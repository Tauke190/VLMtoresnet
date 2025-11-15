import os
import time
import random
import torch
import numpy as np
from torchvision import models, transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.optim.lamb import Lamb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TRAINING_FRACTION = 0.01  # Use 20% of images per class for training

def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}h:{m:02d}m:{s:02d}s"

def main():
    # ==== Paths ====
    TRAIN_DIR = os.path.expanduser('/home/c3-0/datasets/ImageNet/train')
    VAL_DIR = os.path.expanduser('/home/c3-0/datasets/ImageNet/validation')

    # ==== Config ====
    BATCH_SIZE = 256
    EPOCHS = 600  # Increase if you want the 1000-epoch estimate to trigger
    LR = 5e-3
    WEIGHT_DECAY = 0.01
    NUM_CLASSES = 1000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # ==== Data Augmentation ====
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=7, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.95)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ==== Dataset & DataLoader ====
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # Use only 20% of images per class for training
    from collections import defaultdict

    targets = np.array(train_dataset.targets)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    selected_indices = []
    for label, idxs in class_indices.items():
        n_select = max(1, int(TRAINING_FRACTION * len(idxs)))
        selected = np.random.choice(idxs, n_select, replace=False)
        selected_indices.extend(selected.tolist())

    train_subset = torch.utils.data.Subset(train_dataset, selected_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True
    )

    total_batches_per_epoch = len(train_loader)
    total_batches_all_epochs = total_batches_per_epoch * EPOCHS

    print("Initializing model...")
    model = models.resnet50(weights=None, num_classes=NUM_CLASSES).to(DEVICE)

    # ==== Optimizer & Scheduler ====
    optimizer = Lamb(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ==== Loss & Mixup ====
    criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, num_classes=NUM_CLASSES)

    # ==== EMA ====
    ema = ModelEma(model, decay=0.9999, device=DEVICE)

    # ==== AMP ====
    scaler = GradScaler()

    # ==== Validation ====
    def evaluate(model, dataloader, device, topk=(1, 5)):
        model.eval()
        top1_correct = 0.0
        top5_correct = 0.0
        total = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(images)
                _, pred = outputs.topk(max(topk), 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                top1_correct += correct[:1].reshape(-1).float().sum(0).item()
                top5_correct += correct[:5].reshape(-1).float().sum(0).item()
                total += targets.size(0)
        top1 = 100. * top1_correct / max(total, 1)
        top5 = 100. * top5_correct / max(total, 1)
        return top1, top5

    # ==== Training Loop ====
    global_start = time.time()
    best_top1 = -1.0
    epochs_no_improve = 0
    PATIENCE = 10

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        accum_batches = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                epoch0_start_time = time.time()

            images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            images, targets = mixup_fn(images, targets)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            running_loss += loss.item()
            accum_batches += 1

            if batch_idx == 0:
                print(f"[Epoch {epoch+1} Batch 1] Loss: {loss.item():.4f}")

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / accum_batches
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss (last {accum_batches}): {avg_loss:.4f}")
                running_loss = 0.0
                accum_batches = 0

            # Time estimate after first 5 batches of first epoch
            if epoch == 0 and (batch_idx + 1) == 100:
                elapsed_5 = time.time() - epoch0_start_time
                avg_batch_5 = elapsed_5 / 100
                est_total_time_5 = avg_batch_5 * total_batches_all_epochs
                remaining_5 = est_total_time_5 - elapsed_5
                print(f"[Time Estimate @5 batches] Avg batch: {avg_batch_5:.4f}s | "
                      f"Est total: {format_seconds(est_total_time_5)} | "
                      f"Remaining: {format_seconds(remaining_5)}")

            # Time estimate after first 10 batches of first epoch
            if epoch == 0 and (batch_idx + 1) == 1000:
                elapsed_10 = time.time() - epoch0_start_time
                avg_batch_10 = elapsed_10 / 1000.0
                est_total_time_10 = avg_batch_10 * total_batches_all_epochs
                remaining_10 = est_total_time_10 - elapsed_10
                print(f"[Time Estimate @10 batches] Avg batch: {avg_batch_10:.4f}s | "
                      f"Est total: {format_seconds(est_total_time_10)} | "
                      f"Remaining: {format_seconds(remaining_10)}")

        scheduler.step()

        top1, top5 = evaluate(ema.ema, val_loader, DEVICE)
        print(f"[Epoch {epoch+1}] Validation Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

        # Early stopping check (based on Top-1)
        if top1 > best_top1:
            best_top1 = top1
            epochs_no_improve = 0
            best_ckpt_path = f"{script_name}_best.pth"
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'top1': top1}, best_ckpt_path)
            print(f"New best Top-1 {top1:.2f}% — saved: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered (no Top-1 improvement for {PATIENCE} epochs).")
                break

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
        }
        checkpoint_path = f"{script_name}_epoch{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Epoch timing and ETA (includes training, validation, and checkpoint I/O)
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - global_start
        completed_epochs = epoch + 1
        avg_epoch_time = elapsed_total / completed_epochs
        remaining_epochs = EPOCHS - completed_epochs
        est_remaining_time = avg_epoch_time * remaining_epochs
        print(f"[Epoch {epoch+1}] Time: {format_seconds(epoch_time)} | "
              f"Elapsed: {format_seconds(elapsed_total)} | "
              f"ETA: {format_seconds(est_remaining_time)} "
              f"(avg/epoch {avg_epoch_time:.2f}s)")

if __name__ == "__main__":
    main()