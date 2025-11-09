import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.optim.lamb import Lamb

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    print("Initializing model...")
    model = models.resnet50(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)

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
            for i, (images, targets) in enumerate(dataloader):
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
    epoch_durations = []
    global_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
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

            if (batch_idx + 1) == 1:
                avg_loss = running_loss
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Loss: {avg_loss:.4f}")
                running_loss = 0.0

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

        scheduler.step()

        top1, top5 = evaluate(ema.module, val_loader, DEVICE)
        print(f"[Epoch {epoch+1}] Validation Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

        epoch_time = time.time() - epoch_start
        epoch_durations.append(epoch_time)

        # Estimate after 100 epochs
        if (epoch + 1) == 100:
            avg_100 = sum(epoch_durations[:100]) / 100.0
            est_total_100 = avg_100 * EPOCHS
            elapsed_so_far = time.time() - global_start
            remaining_100 = est_total_100 - elapsed_so_far
            print(f"[Epoch 100 Time Estimate] Avg epoch time: {avg_100:.2f}s | "
                  f"Est total: {format_seconds(est_total_100)} | "
                  f"Remaining: {format_seconds(remaining_100)}")

        # Estimate after 1000 epochs (only if reached)
        if (epoch + 1) == 1000:
            avg_1000 = sum(epoch_durations[:1000]) / 1000.0
            est_total_1000 = avg_1000 * EPOCHS
            elapsed_so_far = time.time() - global_start
            remaining_1000 = est_total_1000 - elapsed_so_far
            print(f"[Epoch 1000 Time Estimate] Avg epoch time: {avg_1000:.2f}s | "
                  f"Est total: {format_seconds(est_total_1000)} | "
                  f"Remaining: {format_seconds(remaining_1000)}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
        }
        checkpoint_path = f"{script_name}_epoch{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()