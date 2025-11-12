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
from pathlib import Path
import sys
import os

# --- Configuration (copied from finalfeature_..., except distillation strategy remains CRD+MSE) ---
TRAIN_SUBSET_RATIO = 0.2
TRAIN_EVAL_WITHIN_SUBSET_RATIO = 0.05  # used for validation accuracy in each epoch

TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'
# TRAIN_DIR = '~/data/datasets/imagenet/train'
# VAL_DIR = '~/data/datasets/imagenet/val'

BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Early stopping (copied)
VAL_ACC_DROP_THRESHOLD = 10.0  # Early stopping if val accuracy drops by more than this %

# New eval config (copied)
EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'  # ImageFolder layout

# --- CRD configuration (keep original distillation strategy) ---
USE_CRD = True
CRD_WEIGHT = 0.9  # weight for contrastive distillation loss term relative to other losses

# Paths and utils
PROJECT_ROOT = Path(__file__).parent
CLIP_DIR = PROJECT_ROOT / "CLIP"
TEMPLATES_DIR = CLIP_DIR / "dataloaders" / "templates"
sys.path.append(str(CLIP_DIR))
sys.path.append(str(PROJECT_ROOT))

from utils import (
    zeroshot_classifier,
    get_teacher_features,
    get_student_features,
    imagenet_aligned_classnames,
    imagefolder_human_names,
    read_txt,
    save_checkpoint,
    compute_flops,
    plot_and_save_losses,
)

def evaluate_zero_shot(backbone, projector, loader, zs_weights, device=DEVICE):
    backbone.eval()
    projector.eval()
    zs_weights = zs_weights.to(device=device, dtype=torch.float32)

    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            student_feats = get_student_features(backbone, images)
            proj_feats = projector(student_feats).float()
            proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)
            logits = 100.0 * (proj_feats @ zs_weights)
            _, top5 = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5[:, 0] == labels).sum().item()
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5

# Dataloader/build pipeline (copied)
def build_imagenet_loaders(
    train_dir,
    val_dir,
    transform,
    batch_size=32,
    subset_ratio=0.2,
    eval_ratio_within_subset=0.2,
    num_workers=2,
    seed=42,
):
    train_dir = os.path.expanduser(train_dir)
    val_dir = os.path.expanduser(val_dir)

    base_train = ImageFolder(root=train_dir, transform=transform)
    base_val = ImageFolder(root=val_dir, transform=transform)

    print(f"Loaded ImageNet train from: {train_dir} ({len(base_train)} images, {len(base_train.classes)} classes)")
    print(f"Loaded ImageNet val   from: {val_dir} ({len(base_val)} images, {len(base_val.classes)} classes)")

    # Stratified per-class subset from train
    class_to_indices = {}
    for idx, cls in enumerate(base_train.targets):
        class_to_indices.setdefault(cls, []).append(idx)

    g = torch.Generator().manual_seed(seed)
    selected_indices = []
    for cls, idxs in class_to_indices.items():
        k_subset = max(1, int(len(idxs) * subset_ratio))
        perm = torch.randperm(len(idxs), generator=g).tolist()
        selected = [idxs[p] for p in perm[:k_subset]]
        selected_indices.extend(selected)

    # Split the selected subset into train/eval (stratified, per-class)
    train_split_indices, eval_split_indices = [], []
    selected_set = set(selected_indices)
    for cls, idxs in class_to_indices.items():
        selected_cls = [i for i in idxs if i in selected_set]
        if not selected_cls:
            continue
        perm = torch.randperm(len(selected_cls), generator=g).tolist()
        selected_cls = [selected_cls[p] for p in perm]

        if len(selected_cls) == 1:
            train_split_indices.extend(selected_cls)
            continue

        k_eval = max(1, int(round(len(selected_cls) * eval_ratio_within_subset)))
        k_eval = min(k_eval, len(selected_cls) - 1)
        eval_split_indices.extend(selected_cls[:k_eval])
        train_split_indices.extend(selected_cls[k_eval:])

    train_subset = Subset(base_train, train_split_indices)
    eval_subset = Subset(base_train, eval_split_indices)

    print(
        f"Using train subset (for training): {len(train_split_indices)} images "
        f"and train-eval subset (for validation during training): {len(eval_split_indices)} images "
        f"(subset_ratio={subset_ratio:.2f}, eval_within_subset={eval_ratio_within_subset:.2f})"
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    full_val_loader = DataLoader(base_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, train_eval_loader, full_val_loader, base_train, base_val

# CRD loss (keep original)
def contrastive_distill_loss(student_features_norm, labels, class_text_features_norm, logit_scale=None):
    # student_features_norm: [B, D], normalized
    # class_text_features_norm: [C, D], normalized
    logits = student_features_norm @ class_text_features_norm.t()  # [B, C]
    if logit_scale is not None:
        logits = logits * logit_scale.exp()
    ce = nn.CrossEntropyLoss()
    return ce(logits, labels)

def run_distillation():
    print(f"Using device: {DEVICE}")

    # --- Setup Models ---
    print("Loading teacher model (CLIP ViT-L/14)...")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Loading student model (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim
    student_feature_dim = backbone.num_features

    print("Computing FLOPs and parameters for the student model...")
    compute_flops(backbone, resolution=(3, 224, 224))

    # --- Datasets and loaders (copied structure) ---
    print("Building datasets and dataloaders (subset train, small train-eval split, full val)...")
    train_loader, train_eval_loader, val_loader, base_train, base_val = build_imagenet_loaders(
        TRAIN_DIR,
        VAL_DIR,
        transform=preprocess,
        batch_size=BATCH_SIZE,
        subset_ratio=TRAIN_SUBSET_RATIO,
        eval_ratio_within_subset=TRAIN_EVAL_WITHIN_SUBSET_RATIO,
        num_workers=2,
        seed=42,
    )
    num_classes = len(base_train.classes)
    print(f"Found {num_classes} classes in ImageNet.")

    try:
        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
        distill_loss_fn = nn.MSELoss()
        params_to_train = list(backbone.parameters()) + list(projector.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        # Templates and zero-shot weights (copied pattern; aligned per dataset)
        imagenet_templates = read_txt(str(TEMPLATES_DIR / "imagenet1k.txt"))
        print("Loaded ImageNet templates from CLIP/dataloaders/templates/imagenet1k.txt")
        print(f"Templates count: {len(imagenet_templates)}")

        print("Building ImageNet zero-shot weights for train-eval loader...")
        imagenet_class_names_train = imagenet_aligned_classnames(base_train, "imagenet_class_index.json")
        imagenet_zs_weights_train = zeroshot_classifier(
            imagenet_class_names_train, imagenet_templates, teacher
        ).to(DEVICE)  # [D, C_train]
        # For CRD anchors (class text features): [C, D], normalized
        text_features_train = imagenet_zs_weights_train.t().contiguous()
        text_features_train = text_features_train / text_features_train.norm(dim=-1, keepdim=True)

        print("Building ImageNet zero-shot weights for full val loader...")
        imagenet_class_names_val = imagenet_aligned_classnames(base_val, "imagenet_class_index.json")
        imagenet_zs_weights_val = zeroshot_classifier(
            imagenet_class_names_val, imagenet_templates, teacher
        ).to(DEVICE)  # [D, C_val]

        # Optional: Oxford-IIIT Pet zero-shot eval
        if EVAL_OXFORD_PET:
            print(f"Loading Oxford-IIIT Pet validation dataset from: {OXFORD_PET_VAL_DIR}")
            pet_val_dataset = ImageFolder(root=os.path.expanduser(OXFORD_PET_VAL_DIR), transform=preprocess)
            pet_val_loader = DataLoader(pet_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            pet_class_names = imagefolder_human_names(pet_val_dataset)
            pet_templates = read_txt(str(TEMPLATES_DIR / "pets.txt"))
            print(f"Loaded Pet templates from CLIP/dataloaders/templates/pets.txt (count={len(pet_templates)})")
            print("Building Pet zero-shot weights...")
            pet_zs_weights = zeroshot_classifier(pet_class_names, pet_templates, teacher).to(DEVICE)
        else:
            pet_val_loader, pet_zs_weights = None, None

        # Optional learnable/fixed logit scale (keep original)
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07)).to(DEVICE)

        print("\nStarting logit distillation + CRD...")
        total_start_time = time.time()

        scaler = GradScaler()

        # Initial zero-shot evaluation on the small held-out train-eval subset
        print("\nInitial zero-shot evaluation (student projected into CLIP space):")
        top1, top5 = evaluate_zero_shot(backbone, projector, train_eval_loader, imagenet_zs_weights_train, DEVICE)
        print(f"[Train-Eval SUBSET] Initial Zero-shot: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Initial Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")

        # Save initial checkpoint
        save_checkpoint(backbone, projector, 0, PROJECT_ROOT, __file__)

        train_losses = []
        val_accuracies = []
        best_val_acc = None  # for early stopping based on top1 on train-eval subset

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()

            backbone.train()
            projector.train()
            running_loss = 0.0

            first_100_time = 0.0
            first_1000_time = 0.0
            eta_100_printed = False
            eta_1000_printed = False

            for i, (images, labels) in enumerate(train_loader):
                batch_start = time.time()

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    # Teacher image features for MSE distillation
                    teacher_features = get_teacher_features(teacher, images).float()
                    student_features = get_student_features(backbone, images)
                    projected_student_features = projector(student_features)

                    # Normalize for cosine space losses
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                    projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)

                    # MSE feature distillation
                    final_feature_loss = nn.MSELoss()(projected_student_features, teacher_features)

                    # CRD
                    if USE_CRD:
                        loss_crd = contrastive_distill_loss(
                            projected_student_features, labels, text_features_train, logit_scale=logit_scale
                        )
                    else:
                        loss_crd = torch.tensor(0.0, device=DEVICE)

                    # Total loss (keep original CRD combo)
                    total_loss = 0.1 * final_feature_loss + CRD_WEIGHT * loss_crd

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += total_loss.item()

                # ETA logging (first epoch only)
                if epoch == 0:
                    batch_time = time.time() - batch_start
                    if i < 100:
                        first_100_time += batch_time
                        if i == 99 and not eta_100_printed:
                            avg_bt_100 = first_100_time / 100
                            total_batches_all_epochs = NUM_EPOCHS * len(train_loader)
                            est_total_seconds_100 = avg_bt_100 * total_batches_all_epochs
                            est_hours_100 = est_total_seconds_100 / 3600
                            est_minutes_per_epoch_100 = (avg_bt_100 * len(train_loader)) / 60
                            print(f"[ETA-100] Avg batch {avg_bt_100*1000:.1f} ms -> "
                                  f"Estimated total training time: {est_hours_100:.2f} h "
                                  f"(~{est_minutes_per_epoch_100:.1f} min/epoch based on first 100 batches).")
                            eta_100_printed = True
                    if i < 1000:
                        first_1000_time += batch_time
                        if i == 999 and not eta_1000_printed:
                            avg_bt_1000 = first_1000_time / 1000
                            total_batches_all_epochs = NUM_EPOCHS * len(train_loader)
                            est_total_seconds_1000 = avg_bt_1000 * total_batches_all_epochs
                            est_hours_1000 = est_total_seconds_1000 / 3600
                            est_minutes_per_epoch_1000 = (avg_bt_1000 * len(train_loader)) / 60
                            print(f"[ETA-1000] Avg batch {avg_bt_1000*1000:.1f} ms -> "
                                  f"Estimated total training time: {est_hours_1000:.2f} h "
                                  f"(~{est_minutes_per_epoch_1000:.1f} min/epoch based on first 1000 batches).")
                            eta_1000_printed = True

                if (i + 1) % 100 == 0:
                    avg_loss_so_far = running_loss / (i + 1)
                    if USE_CRD:
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{len(train_loader)}] Avg Loss: {avg_loss_so_far:.4f} (MSE+CRD)")
                    else:
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{len(train_loader)}] Avg Loss: {avg_loss_so_far:.4f}")

            epoch_loss = running_loss / len(train_loader)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")
            train_losses.append(epoch_loss)

            epoch_time = time.time() - epoch_start_time
            print(f"Time taken for Epoch {epoch+1}: {epoch_time / 60:.2f} minutes")

            # Evaluate on the held-out split from the train subset
            if EVAL_FULL_VAL_EACH_EPOCH:
                top1, top5 = evaluate_zero_shot(backbone, projector, train_eval_loader, imagenet_zs_weights_train, DEVICE)
                print(f"[Train-Eval SUBSET] Zero-shot after Epoch {epoch+1}: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
                val_accuracies.append(top1)
                if EVAL_OXFORD_PET and pet_val_loader is not None:
                    pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
                    print(f"[Oxford-Pet] Zero-shot after Epoch {epoch+1}: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")

            # Update plot
            plot_and_save_losses(train_losses, val_accuracies, __file__, fig_title="Contrastive distillation")

            # Early stopping (copied logic): stop if validation Top-1 drops > threshold; save best
            if best_val_acc is None:
                best_val_acc = top1
                save_checkpoint(backbone, projector, epoch + 1, PROJECT_ROOT, __file__)
            else:
                if top1 < best_val_acc - VAL_ACC_DROP_THRESHOLD:
                    print(f"Early stopping triggered: validation accuracy dropped by more than {VAL_ACC_DROP_THRESHOLD:.1f}% (from {best_val_acc:.2f}% to {top1:.2f}%).")
                    break
                if top1 > best_val_acc:
                    best_val_acc = top1
                    save_checkpoint(backbone, projector, epoch + 1, PROJECT_ROOT, __file__)

        print("---------------------------------")

        # Final evaluation on full ImageNet val
        print("\nFinal validation:")
        top1, top5 = evaluate_zero_shot(backbone, projector, val_loader, imagenet_zs_weights_val, DEVICE)
        print(f"[ImageNet FULL] Final Zero-shot: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Final Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")
        print("\nDistillation training finished.")
        total_time = time.time() - total_start_time
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        plot_and_save_losses(train_losses, val_accuracies, __file__)

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

if __name__ == '__main__':
    run_distillation()