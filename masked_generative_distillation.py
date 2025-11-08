# Logit distillation + Supervised finetuning + Masked Generative Distillation (MGD)
# Description:
# A script to perform knowledge distillation from a CLIP ViT-L/14 teacher to a
# ResNet-50 student, with validation on a subset of the validation set
# after each epoch. This version includes:
# - Final feature (global) MSE distillation in the CLIP joint space
# - Zero-shot evaluation with CLIP text embeddings (ImageNet subset + Oxford-IIIT Pet)
# - Masked Generative Distillation (MGD) using teacher ViT deep tokens
#
# MGD (ViTKD-style Generation for Deep Layers):
# 1) Align student deep feature map to teacher token width with 1x1 conv
# 2) Resize to teacher token grid (e.g., 16x16 for ViT-L/14 @ 224)
# 3) Randomly mask a ratio of tokens and replace them with a learnable masked token
# 4) Feed masked features through a small generative block (conv projector)
# 5) Compute MSE only on masked positions against teacher tokens
#
# Dependencies:
# pip install torch torchvision timm git+https://github.com/openai/CLIP.git thop

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
import clip
import time
import random
import math
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path
from typing import Tuple

# --- Configuration ---
TRAIN_SUBSET_RATIO = 0.15

# TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
# VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'
TRAIN_DIR = '~/data/datasets/imagenet/train'
VAL_DIR = '~/data/datasets/imagenet/val'

VAL_SUBSET_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

# New eval config
EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'  # ImageFolder layout

# --- MGD configuration ---
USE_MGD = True
MGD_MASK_RATIO = 0.4          # lambda: ratio of tokens to mask (0..1)
MGD_WEIGHT = 1.0              # weight for MGD loss
MGD_CONV_EXPANSION = 1.0      # keep same channels inside generative block
PRINT_ETA_AT = (100, 1000)    # batches at which we print ETA

def get_teacher_features(model, images):
    # CLIP joint-space pooled embedding (normalized later)
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(backbone, images):
    # Global pooled features
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    return pooled_features

def load_prompts_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            templates = [line.strip() for line in f.readlines()]
        if len(templates) == 0:
            templates = ["a photo of a {}"]
        print(f"Loaded {len(templates)} templates from {filepath}.")
        return templates
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}. Using a default template.")
        return ["a photo of a {}"]

# Replace zeroshot_validate_student with a version that uses precomputed text features
def zeroshot_validate_student(backbone, projector, val_loader, text_features, logit_scale=None, device=DEVICE):
    text_features = text_features.to(device)
    top1_correct, top5_correct, total = 0, 0, 0
    backbone.eval()
    projector.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = get_student_features(backbone, images)
            student_features = projector(features)
            student_features = student_features / student_features.norm(dim=-1, keepdim=True)
            logits = student_features @ text_features.t()
            if logit_scale is not None:
                logits = logits * logit_scale.exp()
            _, top5_preds = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy

# Compute FLOPs on CPU to save VRAM
def compute_flops(model, resolution=(3, 224, 224)):
    from thop import profile
    model_cpu = model.to('cpu')
    dummy = torch.randn(1, *resolution)
    with torch.no_grad():
        flops, params = profile(model_cpu, inputs=(dummy,), verbose=False)
    model.to(DEVICE)
    print(f"\n\n***** FLOP TOTAL: {flops / 10 ** 9:.2f} GFLOPs *****")
    print(f"***** Model Parameters: {params:,} *****\n")
    return flops / 10 ** 9, params

def load_imagenet_classnames(json_path="imagenet_class_index.json"):
    with open(json_path, "r") as f:
        idx_to_data = json.load(f)
    # Returns list indexed by class idx: human-readable name
    return [idx_to_data[str(i)][1].replace('_', ' ') for i in range(len(idx_to_data))]

# New: map synset -> readable name, then align to ImageFolder class order
def load_imagenet_synset_to_name(json_path="imagenet_class_index.json"):
    with open(json_path, "r") as f:
        idx_to_data = json.load(f)
    # idx -> (synset, name)
    return {idx_to_data[str(i)][0]: idx_to_data[str(i)][1].replace('_', ' ') for i in range(len(idx_to_data))}

def imagenet_aligned_classnames(dataset, json_path="imagenet_class_index.json"):
    syn_to_name = load_imagenet_synset_to_name(json_path)
    return [syn_to_name.get(syn, syn) for syn in dataset.classes]

def imagefolder_human_names(dataset):
    return [c.replace('_', ' ') for c in dataset.classes]

# New: precompute text features in chunks (avoids OOM) and cache
def precompute_text_features(teacher, class_names, templates, device=DEVICE, chunk_size=256):
    teacher.eval()
    all_cls_feats = []
    with torch.no_grad():
        for name in class_names:
            feats_chunks = []
            for i in range(0, len(templates), chunk_size):
                batch_prompts = [templates[j].format(name) for j in range(i, min(i + chunk_size, len(templates)))]
                tokens = clip.tokenize(batch_prompts).to(device)
                feats = teacher.encode_text(tokens).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_chunks.append(feats)
            cls_feats = torch.cat(feats_chunks, dim=0).mean(dim=0, keepdim=True)
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
            all_cls_feats.append(cls_feats)
    return torch.cat(all_cls_feats, dim=0)  # [num_classes, d]

# Custom prompt templates for Oxford-IIIT Pet
OXFORD_PET_PROMPTS = [
    "a photo of a {}",
    "a photo of a {}, a type of pet",
    "a photo of big {}",
    "a photo of a small {}"
]

# --------- MGD helpers ---------

def get_vit_token_width_from_visual(visual) -> int:
    """
    Robustly infer the ViT token width (transformer width) from a CLIP visual module
    across different CLIP versions.
    """
    # Some versions expose it directly
    if hasattr(visual, "width"):
        return int(visual.width)
    # Positional embedding: [1, 1+Nt, width]
    if hasattr(visual, "positional_embedding") and visual.positional_embedding is not None:
        return int(visual.positional_embedding.shape[-1])
    # LayerNorm over 'width'
    if hasattr(visual, "ln_post") and hasattr(visual.ln_post, "weight"):
        return int(visual.ln_post.weight.shape[0])
    # Conv1 out_channels equals 'width' for ViT patch embed
    if hasattr(visual, "conv1") and hasattr(visual.conv1, "out_channels"):
        return int(visual.conv1.out_channels)
    raise AttributeError("Cannot infer ViT token width from CLIP visual module.")

def get_teacher_vit_tokens(teacher, images) -> Tuple[torch.Tensor, int, int, int]:
    """
    Extract deep transformer tokens from CLIP ViT visual backbone.
    Returns:
      tokens_no_cls: [B, Nt, Dt] tokens without class token
      Nt: number of tokens (grid size squared)
      Ht, Wt: token grid height and width
    """
    visual = teacher.visual
    B = images.shape[0]
    # follow CLIP-style forward to access tokens
    # conv1 patch embed
    x = visual.conv1(images)  # [B, C, H', W']
    x = x.reshape(B, x.shape[1], -1)
    x = x.permute(0, 2, 1)  # [B, Nt, width]
    # prepend class token
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(B, 1, cls.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)  # [B, 1+Nt, width]
    # positional embedding
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    # transformer expects [seq, batch, dim]
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    # discard class token
    tokens = x[:, 1:, :]  # [B, Nt, Dt]
    Dt = tokens.shape[-1]
    Nt = tokens.shape[1]
    Ht = Wt = int(math.sqrt(Nt))
    return tokens, Nt, Ht, Wt  # tokens in transformer width space (Dt)

class GenerativeBlock(nn.Module):
    """
    Simple conv projector for MGD: preserves channels (teacher token width).
    """
    def __init__(self, channels: int, expansion: float = 1.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def compute_mgd_loss(student_feat_map: torch.Tensor,
                     align_conv: nn.Conv2d,
                     gen_block: nn.Module,
                     masked_token: torch.nn.Parameter,
                     teacher_tokens: torch.Tensor,
                     Ht: int, Wt: int,
                     mask_ratio: float = 0.4) -> torch.Tensor:
    """
    student_feat_map: [B, Cs, Hs, Ws] last student feature map
    align_conv: maps Cs -> Dt (teacher token width)
    gen_block: conv projector
    masked_token: [Dt] learnable vector for masked positions
    teacher_tokens: [B, Nt, Dt] tokens from teacher ViT (no class token)
    Ht, Wt: teacher token grid size
    mask_ratio: fraction of tokens masked
    Returns MSE over masked positions only.
    """
    B, _, Hs, Ws = student_feat_map.shape
    Dt = teacher_tokens.shape[-1]
    Nt = Ht * Wt

    # Align channels to teacher token width
    s = align_conv(student_feat_map)            # [B, Dt, Hs, Ws]
    # Resize to teacher token grid
    s = F.interpolate(s, size=(Ht, Wt), mode='bilinear', align_corners=False)  # [B, Dt, Ht, Wt]

    # Tokens [B, Nt, Dt]
    s_tokens = s.flatten(2).transpose(1, 2).contiguous()

    # Build random mask [B, Nt] with exactly floor(ratio*Nt) positions masked per sample
    num_mask = max(1, int(mask_ratio * Nt))
    mask = torch.zeros(B, Nt, dtype=torch.bool, device=s_tokens.device)
    for b in range(B):
        idx = torch.randperm(Nt, device=s_tokens.device)[:num_mask]
        mask[b, idx] = True

    # Apply masked token
    s_tokens_masked = s_tokens.clone()
    s_tokens_masked[mask] = masked_token.to(s_tokens_masked.dtype)

    # Back to spatial for generative block
    s_masked_map = s_tokens_masked.transpose(1, 2).reshape(B, Dt, Ht, Wt)
    s_gen_map = gen_block(s_masked_map)  # [B, Dt, Ht, Wt]
    s_gen_tokens = s_gen_map.flatten(2).transpose(1, 2).contiguous()  # [B, Nt, Dt]

    # Compute MSE on masked positions only
    # Select masked positions
    pred_masked = s_gen_tokens[mask]          # [num_masked_total, Dt]
    target_masked = teacher_tokens[mask]      # [num_masked_total, Dt]
    loss = F.mse_loss(pred_masked, target_masked)
    return loss

def run_distillation():
    print(f"Using device: {DEVICE}")

    # --- Setup Models ---
    print("Loading teacher model (CLIP ViT-L/14)...")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading student model (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim     # CLIP joint-space dimension (e.g., 768)
    # teacher_token_width may not exist as an attribute on some CLIP versions
    teacher_token_width = get_vit_token_width_from_visual(teacher.visual)  # ViT transformer width
    student_feature_dim = backbone.num_features         # e.g., 2048

    print("Computing FLOPs and parameters for the student model...")
    compute_flops(backbone, resolution=(3, 224, 224))

    train_transform = preprocess
    val_transform = preprocess

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
        base_train = ImageFolder(root=TRAIN_DIR, transform=train_transform)

        # --- Take subset from training set ---
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

        # --- Split subset into train/val (80/20) ---
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
        val_loader_subset = DataLoader(val_subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        print(f"Train subset: {len(train_dataset)} images, Validation subset: {len(val_subset_dataset)} images.")

        print(f"Loading full validation dataset from: {VAL_DIR}")
        full_val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
        val_loader = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Full validation set: {len(full_val_dataset)} images.")

        # Sample a fixed validation subset ONCE
        val_indices_full = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
        val_subset = Subset(full_val_dataset, val_indices_full)
        val_loader_subset = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes in the dataset.")

        # Heads and projectors
        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)  # for zero-shot joint space
        classifier = nn.Linear(student_feature_dim, num_classes).to(DEVICE)         # optional supervised head (kept for completeness)
        distill_loss_fn = nn.MSELoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        # MGD modules
        student_align = nn.Conv2d(student_feature_dim, teacher_token_width, kernel_size=1, bias=True).to(DEVICE)
        gen_block = GenerativeBlock(teacher_token_width, expansion=MGD_CONV_EXPANSION).to(DEVICE)
        masked_token = nn.Parameter(torch.zeros(teacher_token_width, device=DEVICE))
        nn.init.normal_(masked_token, std=0.02)

        params_to_train = list(backbone.parameters()) + \
                          list(projector.parameters()) + \
                          list(classifier.parameters()) + \
                          list(student_align.parameters()) + \
                          list(gen_block.parameters()) + \
                          [masked_token]
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        prompt_file = "prompt/imagenet1k.txt"
        imagenet_class_names_val = imagenet_aligned_classnames(full_val_dataset, "imagenet_class_index.json")
        print("Prepared ImageNet human-readable class names aligned to validation dataset order.")

        templates = load_prompts_from_file(prompt_file)

        # Learnable logit scale for zero-shot (kept frozen unless you add to optimizer)
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07)).to(DEVICE)

        # Precompute text features once (chunked) for ImageNet val
        print("Precomputing ImageNet zero-shot text features (chunked) for validation...")
        text_features_imagenet = precompute_text_features(teacher, imagenet_class_names_val, templates, DEVICE, chunk_size=256)
        torch.cuda.empty_cache()

        # Optional: Oxford-IIIT Pet zero-shot evaluation setup
        if EVAL_OXFORD_PET:
            print(f"Loading Oxford-IIIT Pet validation dataset from: {OXFORD_PET_VAL_DIR}")
            pet_val_dataset = ImageFolder(root=OXFORD_PET_VAL_DIR, transform=val_transform)
            pet_val_loader = DataLoader(pet_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            pet_class_names = imagefolder_human_names(pet_val_dataset)
            print(f"Found {len(pet_class_names)} pet classes.")
            print("Precomputing Pet zero-shot text features (custom prompts)...")
            text_features_pet = precompute_text_features(teacher, pet_class_names, OXFORD_PET_PROMPTS, DEVICE, chunk_size=256)
            torch.cuda.empty_cache()
        else:
            pet_val_loader, text_features_pet = None, None

        print("\nStarting distillation (Global MSE + MGD)...")
        total_start_time = time.time()
        best_loss = float('inf')
        epochs_no_improve = 0

        scaler = GradScaler()

        # Initial zero-shot validation before training
        print("\nInitial zero-shot validation before training:")
        top1, top5 = zeroshot_validate_student(
            backbone, projector, val_loader_subset, text_features_imagenet, logit_scale, DEVICE
        )
        print(f"[ImageNet SUBSET] Initial Zero-shot: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = zeroshot_validate_student(
                backbone, projector, pet_val_loader, text_features_pet, logit_scale, DEVICE
            )
            print(f"[Oxford-Pet] Initial Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()

            backbone.train()
            projector.train()
            classifier.train()
            student_align.train()
            gen_block.train()

            running_loss = 0.0
            batch_times = []

            # Train
            for i, (images, labels) in enumerate(train_loader):
                batch_t0 = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                with autocast():
                    # Teacher features for global (joint-space) distillation
                    teacher_features = get_teacher_features(teacher, images).float()  # [B, D_joint]
                    # Student features and projection to teacher joint dim
                    student_features = get_student_features(backbone, images)        # [B, C_s]
                    projected_student_features = projector(student_features)         # [B, D_joint]

                    # Normalize for cosine-like MSE
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                    projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)

                    # Final feature (global) MSE distillation in normalized space
                    # loss_global = distill_loss_fn(projected_student_features, teacher_features)

                    # MGD: reconstruct teacher deep tokens from masked student aligned tokens
                    if USE_MGD:
                        # Extract teacher deep tokens (no grad)
                        with torch.no_grad():
                            teacher_tokens, Nt, Ht, Wt = get_teacher_vit_tokens(teacher, images)
                            # Teacher tokens are in transformer width (teacher_token_width)
                            # Normalize is optional; use raw token values for reconstruction loss
                        # Student deep feature map (keep grad)
                        student_feat_map = backbone.forward_features(images)  # [B, C_s, Hs, Ws]
                        loss_mgd = compute_mgd_loss(
                            student_feat_map,
                            student_align,
                            gen_block,
                            masked_token,
                            teacher_tokens,
                            Ht, Wt,
                            mask_ratio=MGD_MASK_RATIO
                        )
                    else:
                        loss_mgd = torch.tensor(0.0, device=DEVICE)

                    # Optional supervised CE (kept disabled as in your current code)
                    # logits_cls = classifier(student_features)
                    # loss_ce = ce_loss_fn(logits_cls, labels)

                    # total_loss = loss_global + MGD_WEIGHT * loss_mgd  # + 0.5 * loss_ce (if enabled)
                    total_loss =  MGD_WEIGHT * loss_mgd  # + 0.5 * loss_ce (if enabled)

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(total_loss.detach().item())

                # ETA estimation
                bt = time.time() - batch_t0
                batch_times.append(bt)
                if (i + 1) in PRINT_ETA_AT:
                    avg_time = float(np.mean(batch_times))
                    total_batches = len(train_loader)
                    est_epoch_time = avg_time * total_batches
                    est_total_time = est_epoch_time * NUM_EPOCHS
                    print(f"ETA after {i+1} batches -> per epoch: {est_epoch_time/60:.2f} min, total: {est_total_time/3600:.2f} hr")

                if (i + 1) % 100 == 0:
                    avg_loss_so_far = running_loss / (i + 1)
                    if USE_MGD:
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                              f"Avg Loss: {avg_loss_so_far:.4f} (GlobalMSE+MGD)")
                    else:
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                              f"Avg Loss: {avg_loss_so_far:.4f} (GlobalMSE)")

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
                # ImageNet: only on subset
                top1, top5 = zeroshot_validate_student(
                    backbone, projector, val_loader_subset, text_features_imagenet, logit_scale, DEVICE
                )
                print(f"[ImageNet SUBSET] Zero-shot after Epoch {epoch+1}: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
                # Oxford Pet: on full val set
                if EVAL_OXFORD_PET and pet_val_loader is not None:
                    pet_top1, pet_top5 = zeroshot_validate_student(
                        backbone, projector, pet_val_loader, text_features_pet, logit_scale, DEVICE
                    )
                    print(f"[Oxford-Pet] Zero-shot after Epoch {epoch+1}: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")
            print("---------------------------------")

        # After training: evaluate both on full val sets
        print("\nFinal validation on full sets...")
        top1, top5 = zeroshot_validate_student(
            backbone, projector, val_loader, text_features_imagenet, logit_scale, DEVICE
        )
        print(f"[ImageNet FULL] Final Zero-shot: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_top1, pet_top5 = zeroshot_validate_student(
                backbone, projector, pet_val_loader, text_features_pet, logit_scale, DEVICE
            )
            print(f"[Oxford-Pet] Final Zero-shot: Top-1: {pet_top1:.2f}%, Top-5: {pet_top5:.2f}%")
        print("\nDistillation training finished.")

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

if __name__ == '__main__':
    run_distillation()