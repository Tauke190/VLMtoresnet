# Logit distillation + Supervised finetuning + Masked Generative Distillation (MGD)
# Description:
# Distills CLIP ViT-L/14 teacher into ResNet-50 student:
# - Final feature (global) pathway kept (commented out as in prior version)
# - Zero-shot evaluation now uses zeroshot_classifier + evaluate_zero_shot (no precompute_text_features)
# - Masked Generative Distillation (MGD) reconstructs teacher deep tokens
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
import sys
import os

# --- Configuration ---
TRAIN_SUBSET_RATIO = 0.3

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

EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'


PROJECT_ROOT = Path(__file__).parent
CLIP_DIR = PROJECT_ROOT / "CLIP"
TEMPLATES_DIR = CLIP_DIR / "dataloaders" / "templates"
sys.path.append(str(CLIP_DIR))

# --- MGD configuration ---
USE_MGD = True
MGD_MASK_RATIO = 0.4
MGD_WEIGHT = 1.0
MGD_CONV_EXPANSION = 1.0
PRINT_ETA_AT = (100, 1000)

def get_teacher_features(model, images):
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(backbone, images):
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

def zeroshot_classifier(classnames, templates, model, show_progress=True):
    with torch.no_grad():
        weights = []
        iterator = classnames
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(classnames, desc="Building zero-shot weights", total=len(classnames))
            except Exception:
                iterator = classnames
        for classname in iterator:
            texts = [template.format(classname) for template in templates]
            tokens = clip.tokenize(texts).to(DEVICE)
            class_embeds = model.encode_text(tokens).float()
            class_embeds /= class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed /= class_embed.norm()
            weights.append(class_embed)
        weights = torch.stack(weights, dim=1).to(device=DEVICE, dtype=torch.float32)  # [D, C]
    return weights

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
            logits = 100.0 * (proj_feats @ zs_weights)  # [B, C]
            _, top5 = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5[:, 0] == labels).sum().item()
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5

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
    return [idx_to_data[str(i)][1].replace('_', ' ') for i in range(len(idx_to_data))]

def load_imagenet_synset_to_name(json_path="imagenet_class_index.json"):
    with open(json_path, "r") as f:
        idx_to_data = json.load(f)
    return {idx_to_data[str(i)][0]: idx_to_data[str(i)][1].replace('_', ' ') for i in range(len(idx_to_data))}

def imagenet_aligned_classnames(dataset, json_path="imagenet_class_index.json"):
    syn_to_name = load_imagenet_synset_to_name(json_path)
    return [syn_to_name.get(syn, syn) for syn in dataset.classes]

def imagefolder_human_names(dataset):
    return [c.replace('_', ' ') for c in dataset.classes]


def get_vit_token_width_from_visual(visual) -> int:
    if hasattr(visual, "width"):
        return int(visual.width)
    if hasattr(visual, "positional_embedding") and visual.positional_embedding is not None:
        return int(visual.positional_embedding.shape[-1])
    if hasattr(visual, "ln_post") and hasattr(visual.ln_post, "weight"):
        return int(visual.ln_post.weight.shape[0])
    if hasattr(visual, "conv1") and hasattr(visual.conv1, "out_channels"):
        return int(visual.conv1.out_channels)
    raise AttributeError("Cannot infer ViT token width from CLIP visual module.")

def get_teacher_vit_tokens(teacher, images) -> Tuple[torch.Tensor, int, int, int]:
    visual = teacher.visual
    B = images.shape[0]
    x = visual.conv1(images)
    x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(B, 1, cls.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    tokens = x[:, 1:, :]
    Dt = tokens.shape[-1]
    Nt = tokens.shape[1]
    Ht = Wt = int(math.sqrt(Nt))
    return tokens, Nt, Ht, Wt

class GenerativeBlock(nn.Module):
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
    B, _, Hs, Ws = student_feat_map.shape
    Dt = teacher_tokens.shape[-1]
    Nt = Ht * Wt
    s = align_conv(student_feat_map)
    s = F.interpolate(s, size=(Ht, Wt), mode='bilinear', align_corners=False)
    s_tokens = s.flatten(2).transpose(1, 2).contiguous()
    num_mask = max(1, int(mask_ratio * Nt))
    mask = torch.zeros(B, Nt, dtype=torch.bool, device=s_tokens.device)
    for b in range(B):
        idx = torch.randperm(Nt, device=s_tokens.device)[:num_mask]
        mask[b, idx] = True
    s_tokens_masked = s_tokens.clone()
    s_tokens_masked[mask] = masked_token.to(s_tokens_masked.dtype)
    s_masked_map = s_tokens_masked.transpose(1, 2).reshape(B, Dt, Ht, Wt)
    s_gen_map = gen_block(s_masked_map)
    s_gen_tokens = s_gen_map.flatten(2).transpose(1, 2).contiguous()
    pred_masked = s_gen_tokens[mask]
    target_masked = teacher_tokens[mask]
    loss = F.mse_loss(pred_masked, target_masked)
    return loss

def run_distillation():
    print(f"Using device: {DEVICE}")

    print("Loading teacher model (CLIP ViT-L/14)...")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Loading student model (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim
    teacher_token_width = get_vit_token_width_from_visual(teacher.visual)
    student_feature_dim = backbone.num_features

    print("Computing FLOPs and parameters for the student model...")
    compute_flops(backbone, resolution=(3, 224, 224))

    train_transform = preprocess
    val_transform = preprocess

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
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
        print(f"Using {len(selected_indices)} images (~{TRAIN_SUBSET_RATIO*100:.1f}% per class).")

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
        val_subset_dataset_internal = Subset(trainval_subset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader_subset_internal = DataLoader(val_subset_dataset_internal, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        print(f"Train subset: {len(train_dataset)} images, Internal val subset: {len(val_subset_dataset_internal)} images.")

        print(f"Loading full validation dataset from: {VAL_DIR}")
        full_val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
        val_loader_full = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Full validation set: {len(full_val_dataset)} images.")

        val_indices_fixed = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
        val_subset_fixed = Subset(full_val_dataset, val_indices_fixed)
        val_loader_subset = DataLoader(val_subset_fixed, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes.")

        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
        classifier = nn.Linear(student_feature_dim, num_classes).to(DEVICE)
        distill_loss_fn = nn.MSELoss()
        ce_loss_fn = nn.CrossEntropyLoss()  # (unused)
        student_align = nn.Conv2d(student_feature_dim, teacher_token_width, kernel_size=1, bias=True).to(DEVICE)
        gen_block = GenerativeBlock(teacher_token_width, expansion=MGD_CONV_EXPANSION).to(DEVICE)
        masked_token = nn.Parameter(torch.zeros(teacher_token_width, device=DEVICE))
        nn.init.normal_(masked_token, std=0.02)

        params = list(backbone.parameters()) + \
                 list(projector.parameters()) + \
                 list(classifier.parameters()) + \
                 list(student_align.parameters()) + \
                 list(gen_block.parameters()) + \
                 [masked_token]
        optimizer = optim.AdamW(params, lr=LEARNING_RATE)

        prompt_file = "prompt/imagenet1k.txt"
        imagenet_class_names_val = imagenet_aligned_classnames(full_val_dataset, "imagenet_class_index.json")
        templates = load_prompts_from_file(prompt_file)

        print("Building ImageNet zero-shot classifier weights...")
        imagenet_zs_weights = zeroshot_classifier(imagenet_class_names_val, templates, teacher)

        if EVAL_OXFORD_PET:
            print(f"Loading Oxford-IIIT Pet validation dataset from: {OXFORD_PET_VAL_DIR}")
            pet_val_dataset = ImageFolder(root=OXFORD_PET_VAL_DIR, transform=val_transform)
            pet_val_loader = DataLoader(pet_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            pet_class_names = imagefolder_human_names(pet_val_dataset)

            # Load Pet prompts (fallback to a generic template if file missing)
            pet_prompts = load_prompts_from_file("prompt/oxford_pet.txt")

            print(f"Building Pet zero-shot classifier weights...")
            pet_zs_weights = zeroshot_classifier(pet_class_names, pet_prompts, teacher)
        else:
            pet_val_loader, pet_zs_weights = None, None

        print("\nStarting distillation (MGD only; global MSE kept disabled as before)...")
        best_loss = float('inf')
        epochs_no_improve = 0
        scaler = GradScaler()

        print("\nInitial zero-shot validation:")
        top1_init, top5_init = evaluate_zero_shot(backbone, projector, val_loader_subset, imagenet_zs_weights, DEVICE)
        print(f"[ImageNet SUBSET] Initial Zero-shot: Top-1: {top1_init:.2f}%, Top-5: {top5_init:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_t1_init, pet_t5_init = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Initial Zero-shot: Top-1: {pet_t1_init:.2f}%, Top-5: {pet_t5_init:.2f}%")

        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            backbone.train()
            projector.train()
            classifier.train()
            student_align.train()
            gen_block.train()

            running_loss = 0.0
            batch_times = []

            for i, (images, labels) in enumerate(train_loader):
                bt0 = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    # teacher_features_global = get_teacher_features(teacher, images).float()
                    student_features = get_student_features(backbone, images)
                    projected_student = projector(student_features)
                    projected_student = projected_student / projected_student.norm(dim=-1, keepdim=True)
                    # teacher_features_global = teacher_features_global / teacher_features_global.norm(dim=-1, keepdim=True)
                    # loss_global = distill_loss_fn(projected_student, teacher_features_global)

                    if USE_MGD:
                        with torch.no_grad():
                            teacher_tokens, Nt, Ht, Wt = get_teacher_vit_tokens(teacher, images)
                        student_feat_map = backbone.forward_features(images)
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

                    total_loss = MGD_WEIGHT * loss_mgd  # + loss_global (if enabled)

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(total_loss.detach().item())

                bt = time.time() - bt0
                batch_times.append(bt)
                if (i + 1) in PRINT_ETA_AT:
                    avg_time = float(np.mean(batch_times))
                    total_batches = len(train_loader)
                    est_epoch = avg_time * total_batches
                    est_total = est_epoch * NUM_EPOCHS
                    print(f"ETA after {i+1} batches -> epoch: {est_epoch/60:.2f} min, total: {est_total/3600:.2f} h")

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{len(train_loader)}] "
                          f"Avg Loss: {running_loss/(i+1):.4f}")

            epoch_loss = running_loss / len(train_loader)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Early stopping patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

            print(f"Epoch time: {(time.time() - epoch_start)/60:.2f} min")

            if EVAL_FULL_VAL_EACH_EPOCH:
                top1_e, top5_e = evaluate_zero_shot(backbone, projector, val_loader_subset, imagenet_zs_weights, DEVICE)
                print(f"[ImageNet SUBSET] Zero-shot after Epoch {epoch+1}: Top-1: {top1_e:.2f}%, Top-5: {top5_e:.2f}%")
                if EVAL_OXFORD_PET and pet_val_loader is not None:
                    pet_t1_e, pet_t5_e = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
                    print(f"[Oxford-Pet] Zero-shot after Epoch {epoch+1}: Top-1: {pet_t1_e:.2f}%, Top-5: {pet_t5_e:.2f}%")
            print("---------------------------------")

        print("\nFinal validation on full sets...")
        top1_final, top5_final = evaluate_zero_shot(backbone, projector, val_loader_full, imagenet_zs_weights, DEVICE)
        print(f"[ImageNet FULL] Final Zero-shot: Top-1: {top1_final:.2f}%, Top-5: {top5_final:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_t1_f, pet_t5_f = evaluate_zero_shot(backbone, projector, pet_val_loader, pet_zs_weights, DEVICE)
            print(f"[Oxford-Pet] Final Zero-shot: Top-1: {pet_t1_f:.2f}%, Top-5: {pet_t5_f:.2f}%")
        print("\nDistillation training finished.")

    except FileNotFoundError as e:
        print("Error: Dataset directory not found. Check paths.")
        print(e)
        return

if __name__ == '__main__':
    run_distillation()