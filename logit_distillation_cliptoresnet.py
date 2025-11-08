import os
# Set debug env (must be before init_process_group)
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
# Optional fallback test:
os.environ.setdefault("NCCL_IB_DISABLE", "1")  # enable to avoid IB hangs on non-IB nodes
# os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
import clip
import time
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import copy  # NEW
import multiprocessing as mp  # NEW

# Force 'spawn' to avoid forking after CUDA init deadlocks
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ==== DDP Imports ====
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.optim.lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Configuration ---
TRAIN_SUBSET_RATIO = 0.15
VAL_SUBSET_SIZE = 5000
BATCH_SIZE = 256  # Will be divided by world_size
LEARNING_RATE = 5e-3
NUM_EPOCHS = 600
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4
NUM_WORKERS = 0  # set to 0 to validate hang removal, then raise to 4 with spawn

def setup_ddp():
    import datetime
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=10)
    )
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = torch.cuda.device_count()
    assert local_rank < device_count, f"LOCAL_RANK {local_rank} >= device_count {device_count}"
    torch.cuda.set_device(local_rank)
    print(f"[rank {rank}] init done (local_rank={local_rank}, device_count={device_count})", flush=True)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_teacher_features(model, images):
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(backbone, images):
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    return pooled_features

def validate_student(backbone, projector, teacher, val_loader, device):
    backbone.eval()
    projector.eval()
    teacher.eval()
    total_similarity = 0.0
    total_mse = 0.0
    total = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            student_features = get_student_features(backbone, images)
            projected_student_features = projector(student_features)
            projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)
            teacher_features = teacher.encode_image(images).float()
            teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
            similarity = (projected_student_features * teacher_features).sum(dim=-1)
            total_similarity += similarity.sum().item()
            mse = nn.functional.mse_loss(projected_student_features, teacher_features, reduction='sum')
            total_mse += mse.item()
            total += images.size(0)
    avg_similarity = total_similarity / total
    avg_mse = total_mse / total
    return avg_similarity, avg_mse

def load_prompts_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            templates = [line.strip() for line in f.readlines()]
        return templates
    except FileNotFoundError:
        return []

def zeroshot_validate_student(backbone, projector, class_names, val_loader, teacher, templates, device):
    prompts = [template.format(name) for name in class_names for template in templates]
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = teacher.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    num_templates = len(templates)
    num_classes = len(class_names)
    text_features = text_features.view(num_classes, num_templates, -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    top1_correct = 0
    top5_correct = 0
    total = 0
    backbone.eval()
    projector.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = get_student_features(backbone, images)
            student_features = projector(features)
            student_features = student_features / student_features.norm(dim=-1, keepdim=True)
            logits = student_features @ text_features.t()
            _, top5_preds = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy

def compute_flops(model, resolution=(3, 224, 224), device='cpu'):
    from thop import profile
    input = torch.randn(1, *resolution).to(device)
    flops, params = profile(model, inputs=(input,))
    print(f"\n\n***** FLOP TOTAL: {flops / 10 ** 9:.2f} GFLOPs *****")
    print(f"***** Model Parameters: {params:,} *****\n")
    return flops / 10 ** 9, params

def run_distillation():
    # ==== DDP Setup ====
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    DEVICE = f'cuda:{local_rank}'

    if rank == 0:
        print(f"Distributed training with {world_size} GPUs detected.")

    # --- Setup Models ---
    if rank == 0:
        print("Loading teacher model (CLIP ViT-L/14)...")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    if rank == 0:
        print("Loading student model (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim
    student_feature_dim = backbone.num_features

    if rank == 0:
        print("Computing FLOPs and parameters for the student model...")
        # RUN THOP ON A CPU CLONE SO IT DOESN'T POLLUTE THE TRAINING MODEL WITH CPU BUFFERS
        _model_for_flops = copy.deepcopy(backbone).cpu()
        compute_flops(_model_for_flops, resolution=(3, 224, 224), device='cpu')
        del _model_for_flops
        torch.cuda.empty_cache()
    # (backbone is still on DEVICE and clean here)

    # Remove THOP’s temporary buffers if they were added
    for m in backbone.modules():
        for name in ("total_ops", "total_params"):
            if hasattr(m, name):
                delattr(m, name)
    # And ensure everything is on the correct GPU
    backbone = backbone.to(DEVICE, non_blocking=True)

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
    TRAIN_DIR = os.path.expanduser('~/data/datasets/imagenet/train')
    VAL_DIR = os.path.expanduser('~/data/datasets/imagenet/val')
    base_train = ImageFolder(root=TRAIN_DIR, transform=train_transform)
    targets = base_train.targets
    class_to_indices = {}
    for idx, t in enumerate(targets):
        class_to_indices.setdefault(t, []).append(idx)
    selected_indices = []
    g = torch.Generator().manual_seed(42)
    for cls, idxs in class_to_indices.items():
        k = max(1, int(len(idxs) * TRAIN_SUBSET_RATIO))
        perm = torch.randperm(len(idxs), generator=g)[:k].tolist()
        for p in perm:
            selected_indices.append(idxs[p])
    trainval_subset = Subset(base_train, selected_indices)
    val_ratio_within_subset = 0.20
    num_subset = len(selected_indices)
    num_val = int(num_subset * val_ratio_within_subset)
    num_train = num_subset - num_val
    indices = list(range(num_subset))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    train_dataset = Subset(trainval_subset, train_indices)
    val_subset_dataset = Subset(trainval_subset, val_indices)

    # ==== Distributed Sampler ====
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset_dataset, shuffle=False, drop_last=False)

    # ==== DataLoader ====
    batch_size_per_gpu = BATCH_SIZE // world_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, shuffle=False, NUM_WORKERS=0, pin_memory=True, sampler=train_sampler
    )
    val_loader_subset = DataLoader(
        val_subset_dataset, batch_size=512, shuffle=False, NUM_WORKERS=0, pin_memory=True, sampler=val_sampler
    )

    # ==== DDP Wrap ====
    backbone = DDP(backbone, device_ids=[local_rank], broadcast_buffers=False)
    # Projector and classifier are small, can be left as is or wrapped if needed

    num_classes = len(base_train.classes)
    projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
    classifier = nn.Linear(student_feature_dim, num_classes).to(DEVICE)

    # ==== Optimizer & Scheduler ====
    params_to_train = list(backbone.parameters()) + list(projector.parameters()) + list(classifier.parameters())
    optimizer = Lamb(params_to_train, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ==== Loss, Mixup, EMA ====
    distill_loss_fn = nn.MSELoss()
    criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, num_classes=num_classes)
    ema = ModelEma(backbone.module, decay=0.9999, device=DEVICE)
    scaler = GradScaler()

    prompt_file = "prompt/imagenet1k.txt"
    templates = load_prompts_from_file(prompt_file)
    templates = templates[:2]
    class_names = base_train.classes

    # ==== Training Loop ====
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        backbone.train()
        projector.train()
        classifier.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            images, labels = mixup_fn(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                teacher_features = get_teacher_features(teacher, images).float()
                student_features = get_student_features(backbone.module, images)
                projected_student_features = projector(student_features)
                teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)
                loss_distill = distill_loss_fn(projected_student_features, teacher_features)
                total_loss = loss_distill
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(backbone.module)
            running_loss += total_loss.item()

            if (i + 1) % 100 == 0 and rank == 0:
                avg_loss_so_far = running_loss / (i + 1)
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        if rank == 0:
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if rank == 0:
                print(f"Early stopping patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                if rank == 0:
                    print("Early stopping triggered: training loss has converged.")
                break

        # Validation (only rank 0 prints)
        if rank == 0:
            zeroshot_top1, zeroshot_top5 = zeroshot_validate_student(backbone.module, projector, class_names, val_loader_subset, teacher, templates, DEVICE)
            print(f"Validation Accuracy (Zero-shot) after Epoch {epoch+1}: Top-1: {zeroshot_top1:.5f}%, Top-5: {zeroshot_top5:.5f}%")
            avg_sim, mse_loss = validate_student(backbone.module, projector, teacher, val_loader_subset, DEVICE)
            print(f"Validation (Logits) after Epoch {epoch+1}: Average Similarity: {avg_sim:.5f}, MSE: {mse_loss:.5f}")
            print("---------------------------------")
            checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': backbone.module.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved for epoch {epoch+1}.")
            torch.cuda.empty_cache()

    cleanup_ddp()

if __name__ == '__main__':
    run_distillation()

    # dev = torch.device(DEVICE)
    # for n, t in list(backbone.named_parameters()) + list(backbone.named_buffers()):
    #     if t is not None and (not t.is_cuda or t.device != dev):
    #         raise RuntimeError(f"{n} is on {t.device}, expected {dev}")