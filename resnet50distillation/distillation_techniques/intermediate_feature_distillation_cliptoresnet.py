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

TRAIN_SUBSET_RATIO = 0.2            
TRAIN_EVAL_WITHIN_SUBSET_RATIO = 0.05
# Default dataset paths
# TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
# VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'

TRAIN_DIR = '~/data/datasets/imagenet/train'
VAL_DIR = '~/data/datasets/imagenet/validation'
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Early stopping (borrowed logic from finalfeature script)
VAL_ACC_DROP_THRESHOLD = 10.0  # stop if Top-1 drops by more than this percentage from best
# (Removed train-loss patience early stopping in favor of this)

# Eval config
EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'

# Loss weights (keep original distillation strategy)
FINAL_FEATURE_WEIGHT = 1.0
INTERMEDIATE_FEATURE_WEIGHT = 1.0

# Paths / utils
PROJECT_ROOT = Path(__file__).parent.parent
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
    """
    Evaluates the student model in a zero-shot setting.

    Args:
        backbone (nn.Module): Student backbone model.
        projector (nn.Module): Linear projection head.
        loader (DataLoader): DataLoader for evaluation data.
        zs_weights (torch.Tensor): Zero-shot classifier weights.
        device (str): Device to run evaluation on.

    Returns:
        top1 (float): Top-1 accuracy (%).
        top5 (float): Top-5 accuracy (%).
    """
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

# New: shared loader builder (adapted from finalfeature script)
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
    """
    Builds ImageNet DataLoaders for training, train-eval, and validation splits.

    Args:
        train_dir (str): Path to training data directory.
        val_dir (str): Path to validation data directory.
        transform (callable): Transformations to apply to images.
        batch_size (int): Batch size for DataLoaders.
        subset_ratio (float): Fraction of training data to use.
        eval_ratio_within_subset (float): Fraction of subset for train-eval split.
        num_workers (int): Number of DataLoader workers.
        seed (int): Random seed for reproducibility.

    Returns:
        train_loader (DataLoader): Loader for training subset.
        train_eval_loader (DataLoader): Loader for train-eval subset.
        full_val_loader (DataLoader): Loader for full validation set.
        base_train (ImageFolder): Full training dataset.
        base_val (ImageFolder): Full validation dataset.
    """
    train_dir = os.path.expanduser(train_dir)
    val_dir = os.path.expanduser(val_dir)
    base_train = ImageFolder(root=train_dir, transform=transform)
    base_val = ImageFolder(root=val_dir, transform=transform)

    print(f"Loaded ImageNet train: {len(base_train)} images, {len(base_train.classes)} classes")
    print(f"Loaded ImageNet val:   {len(base_val)} images, {len(base_val.classes)} classes")

    class_to_indices = {}
    for idx, cls in enumerate(base_train.targets):
        class_to_indices.setdefault(cls, []).append(idx)

    g = torch.Generator().manual_seed(seed)
    selected_indices = []
    for cls, idxs in class_to_indices.items():
        k_subset = max(1, int(len(idxs) * subset_ratio))
        perm = torch.randperm(len(idxs), generator=g).tolist()
        selected_cls = [idxs[p] for p in perm[:k_subset]]
        selected_indices.extend(selected_cls)

    train_split_indices, eval_split_indices = [], []
    selected_set = set(selected_indices)
    for cls, idxs in class_to_indices.items():
        cls_selected = [i for i in idxs if i in selected_set]
        if not cls_selected:
            continue
        perm = torch.randperm(len(cls_selected), generator=g).tolist()
        cls_selected = [cls_selected[p] for p in perm]
        if len(cls_selected) == 1:
            train_split_indices.extend(cls_selected)
            continue
        k_eval = max(1, int(round(len(cls_selected) * eval_ratio_within_subset)))
        k_eval = min(k_eval, len(cls_selected) - 1)
        eval_split_indices.extend(cls_selected[:k_eval])
        train_split_indices.extend(cls_selected[k_eval:])

    train_subset = Subset(base_train, train_split_indices)
    eval_subset = Subset(base_train, eval_split_indices)

    print(f"Train subset size: {len(train_split_indices)}; Train-eval subset size: {len(eval_split_indices)} "
          f"(subset_ratio={subset_ratio:.2f}, eval_ratio={eval_ratio_within_subset:.2f})")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    full_val_loader = DataLoader(base_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, train_eval_loader, full_val_loader, base_train, base_val

def register_hook(module, name, feature_dict):
    """
    Registers a forward hook on a module to save its output in a dictionary.

    Args:
        module (nn.Module): Module to register hook on.
        name (str): Key name for feature_dict.
        feature_dict (dict): Dictionary to store features.

    Returns:
        handle: Hook handle for removal.
    """
    def hook_fn(module, input, output):
        if output.dim() == 3 and output.shape[0] > output.shape[1]:
            output = output.permute(1, 0, 2)
        feature_dict[name] = output
    return module.register_forward_hook(hook_fn)

def run_distillation():
    """
    Runs the intermediate feature distillation process.

    Loads teacher and student models, builds dataloaders, performs training with
    feature distillation, evaluates zero-shot accuracy, and saves checkpoints.

    Args:
        None (uses global config and argparse for dataset paths).

    Results:
        Trains student model, prints/logs training/validation metrics, saves checkpoints.
    """
    print(f"Using device: {DEVICE}")

    print("Loading teacher model (CLIP ViT-L/14)...")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Loading student model (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim
    student_feature_dim = backbone.num_features

    print("Computing FLOPs / params for student...")
    compute_flops(backbone, resolution=(3, 224, 224))

    # Build loaders (new logic)
    print("Building ImageNet dataloaders with subset + train-eval split...")
    train_loader, train_eval_loader, full_val_loader, base_train, base_val = build_imagenet_loaders(
        TRAIN_DIR,
        VAL_DIR,
        transform=preprocess,
        batch_size=BATCH_SIZE,
        subset_ratio=TRAIN_SUBSET_RATIO,
        eval_ratio_within_subset=TRAIN_EVAL_WITHIN_SUBSET_RATIO,
        num_workers=2,
        seed=42,
    )


    # Distillation heads
    projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
    student_intermediate_proj = None
    final_feature_loss_fn = nn.MSELoss()
    intermediate_feature_loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(list(backbone.parameters()) + list(projector.parameters()), lr=LEARNING_RATE)
    scaler = GradScaler()

    # Zero-shot weights (train-eval subset ordering and full val ordering)
    imagenet_templates = read_txt(str(TEMPLATES_DIR / "imagenet1k.txt"))
    print("Building zero-shot weights (train-eval subset ordering)...")
    imagenet_class_names_train = imagenet_aligned_classnames(base_train, "imagenet_class_index.json")
    imagenet_zs_weights_train = zeroshot_classifier(imagenet_class_names_train, imagenet_templates, teacher).to(DEVICE)

    print("Building zero-shot weights (full val ordering)...")
    imagenet_class_names_val = imagenet_aligned_classnames(base_val, "imagenet_class_index.json")
    imagenet_zs_weights_val = zeroshot_classifier(imagenet_class_names_val, imagenet_templates, teacher).to(DEVICE)

    # Optional Oxford Pet eval
    if EVAL_OXFORD_PET:
        print(f"Loading Oxford-IIIT Pet validation from: {OXFORD_PET_VAL_DIR}")
        pet_val_dataset = ImageFolder(root=os.path.expanduser(OXFORD_PET_VAL_DIR), transform=preprocess)
        pet_val_loader = DataLoader(pet_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        pet_class_names = imagefolder_human_names(pet_val_dataset)
        pet_templates = read_txt(str(TEMPLATES_DIR / "pets.txt"))
        print("Building Pet zero-shot weights...")
        pet_zs_weights = zeroshot_classifier(pet_class_names, pet_templates, teacher).to(DEVICE)
    else:
        pet_val_loader, pet_zs_weights = None, None

    # Feature dicts + hooks (keep intermediate distillation strategy)
    teacher_features_dict = {}
    student_features_dict = {}
    teacher_handle = register_hook(teacher.visual.transformer.resblocks[5], 'clip_block6', teacher_features_dict)
    student_handle = register_hook(backbone.layer2[1], 'resnet_layer2_1', student_features_dict)

    print("\nStarting final + intermediate feature distillation...")
    train_losses = []
    val_accuracies = []
    best_val_acc = None

    # Initial zero-shot eval
    print("\nInitial zero-shot evaluation (train-eval subset):")
    init_top1, init_top5 = evaluate_zero_shot(backbone, projector, train_eval_loader, imagenet_zs_weights_train, DEVICE)
    print(f"[Train-Eval SUBSET] Initial Zero-shot: Top-1: {init_top1:.2f}%, Top-5: {init_top5:.2f}%")
    if EVAL_OXFORD_PET and pet_val_loader is not None:
        pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
        print(f"[Oxford-Pet] Initial Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")

    save_checkpoint(backbone, projector, 0, PROJECT_ROOT, __file__)
    total_start = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            backbone.train()
            projector.train()
            if student_intermediate_proj is not None:
                student_intermediate_proj.train()

            running_loss = 0.0

            # Timing estimation (first epoch only, similar logic)
            first_100_time, first_1000_time = 0.0, 0.0
            eta_100_printed = False
            eta_1000_printed = False

            for i, (images, labels) in enumerate(train_loader):
                batch_t0 = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                teacher_features_dict.clear()
                student_features_dict.clear()

                with autocast():
                    # Teacher final features
                    teacher_final = get_teacher_features(teacher, images).float()
                    # Student final features
                    student_final = get_student_features(backbone, images)
                    projected_student_final = projector(student_final)

                    # Normalize
                    teacher_final = teacher_final / teacher_final.norm(dim=-1, keepdim=True)
                    projected_student_final = projected_student_final / projected_student_final.norm(dim=-1, keepdim=True)

                    final_loss = final_feature_loss_fn(projected_student_final, teacher_final)

                    # Intermediate feature distillation
                    if 'clip_block6' in teacher_features_dict and 'resnet_layer2_1' in student_features_dict:
                        t_block = teacher_features_dict['clip_block6']     # [B, seq, C_t]
                        teacher_cls = t_block[:, 0].float()                # [B, C_t]

                        s_block = student_features_dict['resnet_layer2_1'] # [B, C_s, H, W]
                        student_pooled = nn.functional.adaptive_avg_pool2d(s_block, (1, 1)).flatten(1)

                        if student_intermediate_proj is None:
                            student_intermediate_proj = nn.Linear(student_pooled.shape[1], teacher_cls.shape[1]).to(DEVICE)
                            optimizer.add_param_group({'params': student_intermediate_proj.parameters()})

                        student_pooled_proj = student_intermediate_proj(student_pooled)

                        # Normalize
                        teacher_cls = teacher_cls / teacher_cls.norm(dim=-1, keepdim=True)
                        student_pooled_proj = student_pooled_proj / student_pooled_proj.norm(dim=-1, keepdim=True)

                        intermediate_loss = intermediate_feature_loss_fn(student_pooled_proj, teacher_cls)
                    else:
                        intermediate_loss = torch.zeros((), device=DEVICE)

                    total_loss = FINAL_FEATURE_WEIGHT * final_loss + INTERMEDIATE_FEATURE_WEIGHT * intermediate_loss

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += total_loss.item()

                if epoch == 0:
                    bt = time.time() - batch_t0
                    if i < 100:
                        first_100_time += bt
                        if i == 99 and not eta_100_printed:
                            avg_bt = first_100_time / 100
                            est_epoch_min = (avg_bt * len(train_loader)) / 60
                            est_total_hours = (avg_bt * len(train_loader) * NUM_EPOCHS) / 3600
                            print(f"[ETA-100] Avg batch {avg_bt*1000:.1f} ms -> ~{est_epoch_min:.1f} min/epoch, {est_total_hours:.2f} h total.")
                            eta_100_printed = True
                    if i < 1000:
                        first_1000_time += bt
                        if i == 999 and not eta_1000_printed:
                            avg_bt = first_1000_time / 1000
                            est_epoch_min = (avg_bt * len(train_loader)) / 60
                            est_total_hours = (avg_bt * len(train_loader) * NUM_EPOCHS) / 3600
                            print(f"[ETA-1000] Avg batch {avg_bt*1000:.1f} ms -> ~{est_epoch_min:.1f} min/epoch, {est_total_hours:.2f} h total.")
                            eta_1000_printed = True

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{len(train_loader)}] "
                          f"Avg Loss: {running_loss / (i + 1):.4f}")

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")
            print(f"Time: {(time.time() - epoch_start)/60:.2f} min")

            # Validation / zero-shot each epoch (train-eval subset)
            if EVAL_FULL_VAL_EACH_EPOCH:
                top1, top5 = evaluate_zero_shot(backbone, projector, train_eval_loader, imagenet_zs_weights_train, DEVICE)
                print(f"[Train-Eval SUBSET] Zero-shot Epoch {epoch+1}: Top-1 {top1:.2f}%, Top-5 {top5:.2f}%")
                val_accuracies.append(top1)

                if EVAL_OXFORD_PET and pet_val_loader is not None:
                    pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
                    print(f"[Oxford-Pet] Zero-shot Epoch {epoch+1}: Top-1 {pet_top1:.2f}%, Top-5 {pet_top5:.2f}%")

            plot_and_save_losses(train_losses, val_accuracies, __file__, fig_title="Intermediate feature distillation")

            # Early stopping logic (drop threshold)
            if best_val_acc is None:
                best_val_acc = top1
                save_checkpoint(backbone, projector, epoch + 1, PROJECT_ROOT, __file__)
            else:
                if top1 < best_val_acc - VAL_ACC_DROP_THRESHOLD:
                    print(f"Early stopping: val Top-1 dropped > {VAL_ACC_DROP_THRESHOLD:.1f}% "
                          f"(from {best_val_acc:.2f} to {top1:.2f}).")
                    break
                if top1 > best_val_acc:
                    best_val_acc = top1
                    save_checkpoint(backbone, projector, epoch + 1, PROJECT_ROOT, __file__)

            print("---------------------------------")

        # Final evaluation on full val
        print("\nFinal evaluation on full ImageNet validation:")
        final_top1, final_top5 = evaluate_zero_shot(backbone, projector, full_val_loader, imagenet_zs_weights_val, DEVICE)
        print(f"[ImageNet FULL] Final Zero-shot: Top-1 {final_top1:.2f}%, Top-5 {final_top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Final Zero-shot: Top-1 {pet_top1:.2f}%, Top-5 {pet_top5:.2f}%")

        total_time = time.time() - total_start
        print(f"\nDistillation finished. Total time: {total_time/60:.2f} min ({total_time/3600:.2f} h)")
        plot_and_save_losses(train_losses, val_accuracies, __file__)

    except FileNotFoundError as e:
        print("Error: dataset directory not found.")
        print(e)
    finally:
        try:
            teacher_handle.remove()
            student_handle.remove()
        except Exception:
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Intermediate Feature Distillation Script")
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR, help='Path to training directory')
    parser.add_argument('--val_dir', type=str, default=VAL_DIR, help='Path to validation directory')
    args = parser.parse_args()
    run_distillation()