# Logit distillation + Supervised finetuning 
# Description:
# A script to perform knowledge distillation from a CLIP ViT-L/14 teacher to a
# ResNet-50 student, with validation on a subset of the validation set
# after each epoch. This version is configured for the Oxford-IIIT Pet Dataset.
#
# Dependencies:
# pip install torch torchvision timm git+https://github.com/openai/CLIP.git

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

# --- Configuration ---
TRAIN_SUBSET_RATIO = 0.2

TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'
# TRAIN_DIR = '~/data/datasets/imagenet/train'
# VAL_DIR = '~/data/datasets/imagenet/val'

VAL_SUBSET_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

# Eval config
EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'  # ImageFolder layout

# Loss weights
FINAL_FEATURE_WEIGHT = 1.0
INTERMEDIATE_FEATURE_WEIGHT = 1.0

# Project paths and utils
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

def register_hook(module, name, feature_dict):
    def hook_fn(module, input, output):
        # For CLIP resblocks, output is typically [seq, batch, width]; standardize to [batch, seq, width]
        if output.dim() == 3 and output.shape[0] > output.shape[1]:
            output = output.permute(1, 0, 2)
        feature_dict[name] = output
    return module.register_forward_hook(hook_fn)

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

    train_transform = preprocess
    val_transform = preprocess

    # --- Register hooks for intermediate features ---
    teacher_features_dict = {}
    student_features_dict = {}

    # Example: 6th transformer block for CLIP, 2nd block of layer2 for ResNet
    teacher_handle = register_hook(teacher.visual.transformer.resblocks[5], 'clip_block6', teacher_features_dict)
    student_handle = register_hook(backbone.layer2[1], 'resnet_layer2_1', student_features_dict)

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
        base_train = ImageFolder(root=os.path.expanduser(TRAIN_DIR), transform=train_transform)

        # --- Take 15% subset from training set (class-balanced) ---
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
        print(f"Using {len(selected_indices)} images (~{TRAIN_SUBSET_RATIO*100:.1f}% per class) from {len(class_to_indices)} classes.")

        # --- Split 15% subset into train/val (80/20) ---
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

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        train_subset_val_loader = DataLoader(val_subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        print(f"Train subset: {len(train_dataset)} images, Validation subset: {len(val_subset_dataset)} images.")

        print(f"Loading full validation dataset from: {VAL_DIR}")
        full_val_dataset = ImageFolder(root=os.path.expanduser(VAL_DIR), transform=val_transform)
        val_loader = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Full validation set: {len(full_val_dataset)} images.")

        # Fixed validation subset for faster eval
        fixed_val_indices = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
        val_subset = Subset(full_val_dataset, fixed_val_indices)
        val_loader_subset = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes in the dataset.")

        # Heads
        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
        student_intermediate_proj = None  # will be created lazily based on observed dims
        distill_loss_fn = nn.MSELoss()
        feature_distill_loss_fn = nn.MSELoss()

        params_to_train = list(backbone.parameters()) + list(projector.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        # Zero-shot weights (ImageNet) and optional Pets
        imagenet_templates = read_txt(str(TEMPLATES_DIR / "imagenet1k.txt"))
        imagenet_class_names = imagenet_aligned_classnames(full_val_dataset, "imagenet_class_index.json")
        print("Building ImageNet zero-shot weights...")
        imagenet_zs_weights = zeroshot_classifier(imagenet_class_names, imagenet_templates, teacher).to(DEVICE)

        if EVAL_OXFORD_PET:
            print(f"Loading Oxford-IIIT Pet validation dataset from: {OXFORD_PET_VAL_DIR}")
            pet_val_dataset = ImageFolder(root=os.path.expanduser(OXFORD_PET_VAL_DIR), transform=val_transform)
            pet_val_loader = DataLoader(pet_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            pet_class_names = imagefolder_human_names(pet_val_dataset)
            pet_templates = read_txt(str(TEMPLATES_DIR / "pets.txt"))
            print(f"Building Pet zero-shot weights (classes={len(pet_class_names)}, templates={len(pet_templates)})...")
            pet_zs_weights = zeroshot_classifier(pet_class_names, pet_templates, teacher).to(DEVICE)
        else:
            pet_val_loader, pet_zs_weights = None, None

        print("\nStarting final+intermediate feature distillation...")
        best_loss = float('inf')
        epochs_no_improve = 0
        scaler = GradScaler()

        # Initial zero-shot evaluation
        print("\nInitial zero-shot evaluation:")
        top1, top5 = evaluate_zero_shot(backbone, projector, val_loader_subset, imagenet_zs_weights, DEVICE)
        print(f"[ImageNet SUBSET] Initial Zero-shot: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Initial Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")

        # Save initial checkpoint
        save_checkpoint(backbone, projector, 0, PROJECT_ROOT, __file__)

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            backbone.train()
            projector.train()
            if student_intermediate_proj is not None:
                student_intermediate_proj.train()

            running_loss = 0.0
            batch_times = []

            for i, (images, labels) in enumerate(train_loader):
                batch_start = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # clear feature caches for hooks
                teacher_features_dict.clear()
                student_features_dict.clear()

                with autocast():
                    # Teacher final features (also triggers teacher hooks)
                    teacher_features = get_teacher_features(teacher, images).float()
                    # Student final features
                    student_features = get_student_features(backbone, images)
                    projected_student_features = projector(student_features)

                    # Normalize final features
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                    projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)

                    # Final feature distillation loss
                    final_feature_loss = distill_loss_fn(projected_student_features, teacher_features)

                    # Intermediate feature distillation
                    # Expect teacher block features as [B, seq, C]; use CLS token at index 0
                    if 'clip_block6' not in teacher_features_dict or 'resnet_layer2_1' not in student_features_dict:
                        # If hooks didn't fire for some reason, skip this batch's intermediate loss
                        intermediate_loss = torch.tensor(0.0, device=DEVICE)
                    else:
                        t_block = teacher_features_dict['clip_block6']  # [B, seq, C]
                        teacher_cls_token = t_block[:, 0].float()  # [B, C_t]

                        s_block = student_features_dict['resnet_layer2_1']  # [B, C_s, H, W]
                        student_pooled = nn.functional.adaptive_avg_pool2d(s_block, (1, 1)).flatten(1)  # [B, C_s]

                        # Create the student intermediate projector lazily and add to optimizer
                        if student_intermediate_proj is None:
                            student_intermediate_proj = nn.Linear(student_pooled.shape[1], teacher_cls_token.shape[1]).to(DEVICE)
                            optimizer.add_param_group({'params': student_intermediate_proj.parameters()})

                        student_pooled_proj = student_intermediate_proj(student_pooled)

                        # Normalize intermediate features
                        teacher_cls_token = teacher_cls_token / teacher_cls_token.norm(dim=-1, keepdim=True)
                        student_pooled_proj = student_pooled_proj / student_pooled_proj.norm(dim=-1, keepdim=True)

                        intermediate_loss = feature_distill_loss_fn(student_pooled_proj, teacher_cls_token)

                    total_loss = FINAL_FEATURE_WEIGHT * final_feature_loss + INTERMEDIATE_FEATURE_WEIGHT * intermediate_loss

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += total_loss.item()

                # ETA prints
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                if i + 1 == 100 or i + 1 == 1000:
                    avg_time = np.mean(batch_times)
                    total_batches = len(train_loader)
                    est_epoch_time = avg_time * total_batches
                    est_total_time = est_epoch_time * NUM_EPOCHS
                    print(f"Estimated time per epoch after {i+1} batches: {est_epoch_time/60:.2f} min")
                    print(f"Estimated total training time after {i+1} batches: {est_total_time/3600:.2f} hr")

                if (i + 1) % 100 == 0:
                    avg_loss_so_far = running_loss / (i + 1)
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.4f} (final+intermediate)")

            epoch_loss = running_loss / len(train_loader)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")

            # Early stopping on train loss
            if epoch_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Early stopping patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered: training loss has converged.")
                    break

            epoch_time = time.time() - epoch_start_time
            print(f"Time taken for Epoch {epoch+1}: {epoch_time / 60:.2f} minutes")

            # Validation after each epoch
            if EVAL_FULL_VAL_EACH_EPOCH:
                top1, top5 = evaluate_zero_shot(backbone, projector, val_loader_subset, imagenet_zs_weights, DEVICE)
                print(f"[ImageNet SUBSET] Zero-shot after Epoch {epoch+1}: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
                if EVAL_OXFORD_PET and pet_val_loader is not None:
                    pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
                    print(f"[Oxford-Pet] Zero-shot after Epoch {epoch+1}: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")

            # Save checkpoint each epoch
            save_checkpoint(backbone, projector, epoch + 1, PROJECT_ROOT, __file__)
            print("---------------------------------")

        # Final validation
        print("\nFinal validation on full sets...")
        top1, top5 = evaluate_zero_shot(backbone, projector, val_loader, imagenet_zs_weights, DEVICE)
        print(f"[ImageNet FULL] Final Zero-shot: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Final Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")
        print("\nDistillation training finished.")

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return
    finally:
        # Remove hooks
        try:
            teacher_handle.remove()
            student_handle.remove()
        except Exception:
            pass

if __name__ == '__main__':
    run_distillation()