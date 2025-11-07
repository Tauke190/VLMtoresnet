import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
from timm.loss import BinaryCrossEntropy
from timm.utils import ModelEma
from timm.optim.lamb import Lamb
import os
import time

# ==== DDP Imports ====
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    # ==== DDP Setup ====
    print("Before DDP setup", flush=True)
    local_rank = setup_ddp()
    print("After DDP setup", flush=True)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Print GPU info (only on rank 0)
    if rank == 0:
        print(f"Distributed training with {world_size} GPUs detected.")

    # ==== Paths ====
    # TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
    # VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'
    TRAIN_DIR = '~/data/datasets/imagenet/train'
    VAL_DIR = '~/data/datasets/imagenet/val'

    # ==== Config ====
    BATCH_SIZE = 256 // world_size  # Split batch across GPUs
    EPOCHS = 600
    LR = 5e-3
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 5
    NUM_CLASSES = 1000  # for ImageNet
    DEVICE = f'cuda:{local_rank}'

    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # ==== Data Augmentation (A1) ====
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

    # ==== Dataset ====
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # ==== Distributed Sampler ====
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    print("Before DataLoader", flush=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True, sampler=val_sampler
    )
    print("After DataLoader", flush=True)

    # ==== Model ====
    model = models.resnet50(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    model = DDP(model, device_ids=[local_rank])

    # ==== Optimizer & Scheduler ====
    optimizer = Lamb(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ==== Loss ====
    criterion = BinaryCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, num_classes=NUM_CLASSES)

    # ==== EMA ====
    ema = ModelEma(model, decay=0.9999, device=DEVICE)

    # ==== AMP ====
    scaler = GradScaler()

    # ==== Validation Function ====
    def evaluate(model, dataloader, device, topk=(1, 5)):
        model.eval()
        top1_correct, top5_correct, total = 0, 0, 0
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, pred = outputs.topk(max(topk), 1, True, True)  # [batch, k]
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                top1_correct += correct[:1].reshape(-1).float().sum(0).item()
                top5_correct += correct[:5].reshape(-1).float().sum(0).item()
                total += targets.size(0)
        top1 = 100. * top1_correct / total
        top5 = 100. * top5_correct / total
        return top1, top5

    # ==== Quick Validation on Subset of Training Data (only on rank 0) ====
    if rank == 0:
        subset_indices = list(range(min(1000, len(train_dataset))))
        subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
        subset_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, sampler=subset_sampler, num_workers=0, pin_memory=True
        )
        print("Running quick validation on a subset of the training data (untrained model)...")
        top1, top5 = evaluate(model, subset_loader, DEVICE)
        print(f"Subset Train Data (Untrained Model) - Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

    # ==== Training Loop ====
    print("Before first batch", flush=True)
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Got batch {batch_idx}", flush=True)
        break  # Just test the first batch

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
        model.train()
        running_loss = 0.0
        batch_start_time = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            images, targets = mixup_fn(images, targets)

            optimizer.zero_grad()
            with autocast():
                print("Before forward", flush=True)
                outputs = model(images)
                print("After forward", flush=True)
                print("Before loss", flush=True)
                loss = criterion(outputs, targets)
                print("After loss", flush=True)
                print("Before backward", flush=True)
                scaler.scale(loss).backward()
                print("After backward", flush=True)
                print("Before step", flush=True)
                scaler.step(optimizer)
                scaler.update()
                print("After step", flush=True)

            ema.update(model)

            running_loss += loss.item()

            if (batch_idx + 1) == 1 and rank == 0:
                avg_loss = running_loss / 100
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

            # Print average loss every 100 batches (only on rank 0)
            if (batch_idx + 1) % 100 == 0 and rank == 0:
                avg_loss = running_loss / 100
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

        scheduler.step()

        # Evaluate on validation set (only on rank 0)
        if rank == 0:
            top1, top5 = evaluate(ema.module, val_loader, DEVICE)
            print(f"[Epoch {epoch+1}] Validation Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

            # ==== Save Checkpoint ====
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
            }
            checkpoint_path = f"{script_name}_epoch{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
