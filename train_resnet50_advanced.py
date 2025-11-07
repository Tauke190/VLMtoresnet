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
    # TRAIN_DIR = '~/data/datasets/imagenet/train'
    # VAL_DIR = '~/data/datasets/imagenet/val'
    TRAIN_DIR = os.path.expanduser('~/data/datasets/imagenet/train')
    VAL_DIR = os.path.expanduser('~/data/datasets/imagenet/val')

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
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)

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
    # criterion = BinaryCrossEntropy()
    criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, num_classes=NUM_CLASSES)

    # ==== EMA ====
    # ema = ModelEma(model, decay=0.9999, device=DEVICE)
    ema = ModelEma(model.module, decay=0.9999, device=DEVICE)

    # ==== AMP ====
    scaler = GradScaler()

    # ==== Validation Function ====
    def evaluate(model, dataloader, device, topk=(1, 5)):
        model.eval()
        top1_correct = 0.0
        top5_correct = 0.0
        total = 0
        print("Starting evaluation...", flush=True)
        with torch.no_grad():
            for i, (images, targets) in enumerate(dataloader):
                print(f"Eval batch {i}", flush=True)
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(images)
                _, pred = outputs.topk(max(topk), 1, True, True)  # [batch, k]
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                top1_correct += correct[:1].reshape(-1).float().sum(0).item()
                top5_correct += correct[:5].reshape(-1).float().sum(0).item()
                total += targets.size(0)
        top1 = 100. * top1_correct / max(total, 1)
        top5 = 100. * top5_correct / max(total, 1)
        return top1, top5

    # ==== Training Loop ====
    # Remove the one-off warm-up iteration that can desync DDP:
    # print("Before first batch", flush=True)
    # for batch_idx, (images, targets) in enumerate(train_loader):
    #     print(f"Got batch {batch_idx}", flush=True)
    #     break

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
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

            # Update EMA with the underlying module
            ema.update(model.module)

            running_loss += loss.item()

            if (batch_idx + 1) == 1 and rank == 0:
                avg_loss = running_loss / 1
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

            if (batch_idx + 1) % 100 == 0 and rank == 0:
                avg_loss = running_loss / 100
                print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

        scheduler.step()

        if rank == 0:
            top1, top5 = evaluate(ema.module, val_loader, DEVICE)
            print(f"[Epoch {epoch+1}] Validation Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
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
