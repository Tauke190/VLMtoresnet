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

# ==== Paths ====
# TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
# VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'

TRAIN_DIR = '~/data/datasets/imagenet/train'
VAL_DIR = '~/data/datasets/imagenet/val'


# ==== Config ====
BATCH_SIZE = 512
EPOCHS = 600
LR = 5e-3
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
NUM_CLASSES = 1000  # for ImageNet
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

# ==== Model ====
model = models.resnet50(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)

# ==== Optimizer & Scheduler ====
optimizer = Lamb(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==== Loss ====
criterion = BinaryCrossEntropy()
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, num_classes=NUM_CLASSES)

# ==== EMA ====
ema = ModelEma(model, decay=0.9999)

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

# ==== Subset for Training Evaluation (fixed at start) ====
subset_size = min(2048, len(train_dataset))
subset_indices = torch.randperm(len(train_dataset))[:subset_size]
subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
subset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=512, sampler=subset_sampler, num_workers=4, pin_memory=True
)

# ==== Training Loop ====
first_100_time, first_1000_time = None, None
total_batches = len(train_loader)
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    batch_start_time = time.time()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        running_loss += loss.item()

        # Timing for ETA estimation
        if batch_idx == 99:
            first_100_time = time.time() - batch_start_time
            est_total_time_100 = (first_100_time / 100) * total_batches * EPOCHS
            print(f"[ETA] Based on first 100 batches: ~{est_total_time_100/3600:.2f} hours for full training")
        if batch_idx == 999:
            first_1000_time = time.time() - batch_start_time
            est_total_time_1000 = (first_1000_time / 1000) * total_batches * EPOCHS
            print(f"[ETA] Based on first 1000 batches: ~{est_total_time_1000/3600:.2f} hours for full training")

        # Print average loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Avg Loss: {avg_loss:.4f}")
            running_loss = 0.0

    scheduler.step()

    # Evaluate on the fixed subset of the training set at the end of each epoch
    top1, top5 = evaluate(ema.module, subset_loader, DEVICE)
    print(f"[Epoch {epoch+1}] Train subset Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Last Batch Loss: {loss.item():.4f}")

    # ==== Save Checkpoint ====
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),    }
    checkpoint_path = f"{script_name}_epoch{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# ==== Validation ====
top1, top5 = evaluate(ema.module, val_loader, DEVICE)
print(f"Validation Top-1 Accuracy: {top1:.2f}%")
print(f"Validation Top-5 Accuracy: {top5:.2f}%")
