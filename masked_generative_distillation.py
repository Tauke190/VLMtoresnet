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
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Tuple
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

EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'

PROJECT_ROOT = Path(__file__).parent
CLIP_DIR = PROJECT_ROOT / "CLIP"
TEMPLATES_DIR = CLIP_DIR / "dataloaders" / "templates"
sys.path.append(str(CLIP_DIR))
sys.path.append(str(PROJECT_ROOT))

# --- Import shared helpers (style copied from finalfeature) ---
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

# --- MGD configuration ---
USE_MGD = True
MGD_MASK_RATIO = 0.4
MGD_WEIGHT = 1.0
MGD_CONV_EXPANSION = 1.0

# New: embedding-level MGD config (on 768-d CLIP image embedding)
USE_EMB_MGD = True
EMB_MGD_MASK_RATIO = 0.4
EMB_MGD_WEIGHT = 1.0
EMB_MGD_MLP_EXP = 2.0
# New: use generator output for zero-shot evaluation
USE_GEN_FOR_ZS = True


def evaluate_zero_shot(backbone, projector, loader, zs_weights, device=DEVICE, logit_scale=100.0, emb_gen_block: nn.Module=None, use_gen: bool=False):
    backbone.eval()
    projector.eval()
    if use_gen and emb_gen_block is not None:
        emb_gen_block.eval()
    zs_weights = zs_weights.to(device=device, dtype=torch.float32)

    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            student_feats = get_student_features(backbone, images)
            proj_feats = projector(student_feats).float()
            proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)
            if use_gen and emb_gen_block is not None:
                # inference: no masking; treat generator as a refinement head
                proj_feats = emb_gen_block(proj_feats)
                proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)
            logits = float(logit_scale) * (proj_feats @ zs_weights)
            _, top5 = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5[:, 0] == labels).sum().item()
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5


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
    Ht = Wt = int((Nt) ** 0.5)
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

    # Align student map to teacher token width and spatial size
    s = align_conv(student_feat_map)
    s = F.interpolate(s, size=(Ht, Wt), mode='bilinear', align_corners=False)
    s_tokens = s.flatten(2).transpose(1, 2).contiguous()  # [B, Nt, Dt]

    # Random mask for each sample
    num_mask = max(1, int(mask_ratio * Nt))
    mask = torch.zeros(B, Nt, dtype=torch.bool, device=s_tokens.device)
    for b in range(B):
        idx = torch.randperm(Nt, device=s_tokens.device)[:num_mask]
        mask[b, idx] = True

    # Replace masked tokens with learned vector
    s_tokens_masked = s_tokens.clone()
    s_tokens_masked[mask] = masked_token.to(s_tokens_masked.dtype)
    s_masked_map = s_tokens_masked.transpose(1, 2).reshape(B, Dt, Ht, Wt)

    # Generate/reconstruct masked tokens
    s_gen_map = gen_block(s_masked_map)
    s_gen_tokens = s_gen_map.flatten(2).transpose(1, 2).contiguous()

    pred_masked = s_gen_tokens[mask]
    target_masked = teacher_tokens[mask]
    loss = F.mse_loss(pred_masked, target_masked)
    return loss
# ----------------------------------------------

class EmbGenerativeBlock(nn.Module):
    """Tiny MLP to reconstruct masked embedding dims"""
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def compute_emb_mgd_loss(projected_student: torch.Tensor,
                         teacher_embed: torch.Tensor,
                         gen_block: nn.Module,
                         masked_dim_vec: torch.nn.Parameter,
                         mask_ratio: float = 0.4) -> torch.Tensor:
    """
    Mask a subset of embedding dimensions of the projected student embedding with a single
    learnable vector and reconstruct masked dims via a small MLP. Loss is computed only
    on masked dimensions against the teacher embedding.
    """
    B, D = projected_student.shape
    num_mask = max(1, int(mask_ratio * D))

    # Build a per-sample boolean mask over dimensions
    mask = torch.zeros(B, D, dtype=torch.bool, device=projected_student.device)
    for b in range(B):
        idx = torch.randperm(D, device=projected_student.device)[:num_mask]
        mask[b, idx] = True

    # Replace masked dims with a single learnable vector value (shared across batch)
    template = masked_dim_vec.view(1, D).expand(B, -1).to(projected_student.dtype)
    s_masked = torch.where(mask, template, projected_student)

    # Reconstruct and supervise only masked dims
    pred = gen_block(s_masked)
    loss = F.mse_loss(pred[mask], teacher_embed[mask])
    return loss


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
    teacher_token_width = get_vit_token_width_from_visual(teacher.visual)
    student_feature_dim = backbone.num_features

    print("Computing FLOPs and parameters for the student model...")
    compute_flops(backbone, resolution=(3, 224, 224))

    train_transform = preprocess
    val_transform = preprocess

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
        base_train = ImageFolder(root=os.path.expanduser(TRAIN_DIR), transform=train_transform)

        # Balanced per-class subset
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

        # Split subset into train/val (80/20)
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
        train_subset_val_dataset = Subset(trainval_subset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        train_subset_val_loader = DataLoader(train_subset_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        print(f"Train subset: {len(train_dataset)} images, Internal val subset: {len(train_subset_val_dataset)} images.")

        print(f"Loading full validation dataset from: {VAL_DIR}")
        full_val_dataset = ImageFolder(root=os.path.expanduser(VAL_DIR), transform=val_transform)
        val_loader_full = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Full validation set: {len(full_val_dataset)} images.")

        # Fixed validation subset for quick eval per epoch
        val_indices_fixed = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
        val_subset_fixed = Subset(full_val_dataset, val_indices_fixed)
        val_loader_subset = DataLoader(val_subset_fixed, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes.")

        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
        # classifier removed; we use projector for zero-shot space
        distill_loss_fn = nn.MSELoss()  # kept if needed later (not used for MGD-only)

        # MGD modules (token-level; keep optional)
        student_align = nn.Conv2d(student_feature_dim, teacher_token_width, kernel_size=1, bias=True).to(DEVICE)
        gen_block = GenerativeBlock(teacher_token_width, expansion=MGD_CONV_EXPANSION).to(DEVICE)
        masked_token = nn.Parameter(torch.zeros(teacher_token_width, device=DEVICE))
        nn.init.normal_(masked_token, std=0.02)

        # New: embedding-level MGD modules
        emb_gen_block = EmbGenerativeBlock(teacher_feature_dim, expansion=EMB_MGD_MLP_EXP).to(DEVICE)
        masked_dim_vec = nn.Parameter(torch.zeros(teacher_feature_dim, device=DEVICE))
        nn.init.normal_(masked_dim_vec, std=0.02)

        params = list(backbone.parameters()) + \
                 list(projector.parameters()) + \
                 list(student_align.parameters()) + \
                 list(gen_block.parameters()) + \
                 [masked_token] + \
                 list(emb_gen_block.parameters()) + \
                 [masked_dim_vec]
        optimizer = optim.AdamW(params, lr=LEARNING_RATE)

        # Templates and zero-shot weights (style from finalfeature)
        imagenet_templates = read_txt(str(TEMPLATES_DIR / "imagenet1k.txt"))
        imagenet_class_names_val = imagenet_aligned_classnames(full_val_dataset, "imagenet_class_index.json")
        print("Loaded ImageNet templates from CLIP/dataloaders/templates/imagenet1k.txt")
        print(f"Templates count: {len(imagenet_templates)}")

        print("Building ImageNet zero-shot weights...")
        imagenet_zs_weights = zeroshot_classifier(imagenet_class_names_val, imagenet_templates, teacher).to(DEVICE)

        if EVAL_OXFORD_PET:
            print(f"Loading Oxford-IIIT Pet validation dataset from: {OXFORD_PET_VAL_DIR}")
            pet_val_dataset = ImageFolder(root=os.path.expanduser(OXFORD_PET_VAL_DIR), transform=val_transform)
            pet_val_loader = DataLoader(pet_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            pet_class_names = imagefolder_human_names(pet_val_dataset)
            pet_templates = read_txt(str(TEMPLATES_DIR / "pets.txt"))
            print(f"Loaded Pet templates from CLIP/dataloaders/templates/pets.txt (count={len(pet_templates)})")
            print("Building Pet zero-shot weights...")
            pet_zs_weights = zeroshot_classifier(pet_class_names, pet_templates, teacher).to(DEVICE)
        else:
            pet_val_loader, pet_zs_weights = None, None

        print("\nStarting distillation (MGD only)...")
        best_loss = float('inf')
        epochs_no_improve = 0
        scaler = GradScaler()

        # Initial zero-shot evaluation
        print("\nInitial zero-shot evaluation (student projected into CLIP space):")
        t_logit_scale = float(teacher.logit_scale.exp().item())
        top1_init, top5_init = evaluate_zero_shot(
            backbone, projector, val_loader_subset, imagenet_zs_weights,
            DEVICE, logit_scale=t_logit_scale, emb_gen_block=emb_gen_block, use_gen=USE_GEN_FOR_ZS
        )
        print(f"[ImageNet SUBSET] Initial Zero-shot: Top-1: {top1_init:.2f}%, Top-5: {top5_init:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_t1_init, pet_t5_init = evaluate_zero_shot(
                backbone, projector, pet_val_loader, pet_zs_weights,
                DEVICE, logit_scale=t_logit_scale, emb_gen_block=emb_gen_block, use_gen=USE_GEN_FOR_ZS
            )
            print(f"[Oxford-Pet] Initial Zero-shot: Top-1: {pet_t1_init:.2f}%, Top-5: {pet_t5_init:.2f}%")

        # Save initial checkpoint
        save_checkpoint(backbone, projector, 0, PROJECT_ROOT, __file__)

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            backbone.train()
            projector.train()
            student_align.train()
            gen_block.train()
            emb_gen_block.train()  # ensure generator trains

            running_loss = 0.0

            # ETA timing like finalfeature
            first_100_time = 0.0
            first_1000_time = 0.0
            eta_100_printed = False
            eta_1000_printed = False

            for i, (images, labels) in enumerate(train_loader):
                batch_start = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                with autocast():
                    # Forward student features
                    student_features = get_student_features(backbone, images)
                    projected_student = projector(student_features)
                    projected_student = projected_student / projected_student.norm(dim=-1, keepdim=True)

                    # New: embedding-level MGD on CLIP image embedding (768-d)
                    if USE_EMB_MGD:
                        with torch.no_grad():
                            t_img = teacher.encode_image(images).float()
                            t_img = F.normalize(t_img, dim=-1)
                        loss_emb_mgd = compute_emb_mgd_loss(
                            projected_student,
                            t_img,
                            emb_gen_block,
                            masked_dim_vec,
                            mask_ratio=EMB_MGD_MASK_RATIO
                        )
                    else:
                        loss_emb_mgd = torch.tensor(0.0, device=DEVICE)

                    # Optional: token-level MGD (original)
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

                    total_loss = (EMB_MGD_WEIGHT * loss_emb_mgd) + (MGD_WEIGHT * loss_mgd)
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(total_loss.detach().item())

                # ETA estimation in first epoch
                if epoch == 0:
                    bt = time.time() - batch_start
                    if i < 100:
                        first_100_time += bt
                        if i == 99 and not eta_100_printed:
                            avg_bt_100 = first_100_time / 100
                            total_batches_all_epochs = NUM_EPOCHS * len(train_loader)
                            est_hours = (avg_bt_100 * total_batches_all_epochs) / 3600
                            est_min_per_epoch = (avg_bt_100 * len(train_loader)) / 60
                            print(f"[ETA-100] Avg batch {avg_bt_100*1000:.1f} ms -> "
                                  f"Estimated total training time: {est_hours:.2f} h "
                                  f"(~{est_min_per_epoch:.1f} min/epoch).")
                            eta_100_printed = True
                    if i < 1000:
                        first_1000_time += bt
                        if i == 999 and not eta_1000_printed:
                            avg_bt_1000 = first_1000_time / 1000
                            total_batches_all_epochs = NUM_EPOCHS * len(train_loader)
                            est_hours = (avg_bt_1000 * total_batches_all_epochs) / 3600
                            est_min_per_epoch = (avg_bt_1000 * len(train_loader)) / 60
                            print(f"[ETA-1000] Avg batch {avg_bt_1000*1000:.1f} ms -> "
                                  f"Estimated total training time: {est_hours:.2f} h "
                                  f"(~{est_min_per_epoch:.1f} min/epoch).")
                            eta_1000_printed = True

                if (i + 1) % 100 == 0:
                    avg_loss_so_far = running_loss / (i + 1)
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{len(train_loader)}] Avg Loss: {avg_loss_so_far:.4f}")

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
                    print("Early stopping triggered.")
                    break

            print(f"Time taken for Epoch {epoch+1}: {(time.time() - epoch_start_time) / 60:.2f} minutes")

            # Per-epoch validation (zero-shot)
            if EVAL_FULL_VAL_EACH_EPOCH:
                top1_e, top5_e = evaluate_zero_shot(
                    backbone, projector, val_loader_subset, imagenet_zs_weights,
                    DEVICE, logit_scale=t_logit_scale, emb_gen_block=emb_gen_block, use_gen=USE_GEN_FOR_ZS
                )
                print(f"[ImageNet SUBSET] Zero-shot after Epoch {epoch+1}: Top-1: {top1_e:.2f}%, Top-5: {top5_e:.2f}%")
                if EVAL_OXFORD_PET and pet_val_loader is not None:
                    pet_t1_e, pet_t5_e = evaluate_zero_shot(
                        backbone, projector, pet_val_loader, pet_zs_weights,
                        DEVICE, logit_scale=t_logit_scale, emb_gen_block=emb_gen_block, use_gen=USE_GEN_FOR_ZS
                    )
                    print(f"[Oxford-Pet] Zero-shot after Epoch {epoch+1}: Top-1: {pet_t1_e:.2f}%, Top-5: {pet_t5_e:.2f}%")

            # Save checkpoint after epoch
            save_checkpoint(backbone, projector, epoch + 1, PROJECT_ROOT, __file__)
            print("---------------------------------")

        print("\nFinal validation on full sets...")
        top1_final, top5_final = evaluate_zero_shot(
            backbone, projector, val_loader_full, imagenet_zs_weights,
            DEVICE, logit_scale=t_logit_scale, emb_gen_block=emb_gen_block, use_gen=USE_GEN_FOR_ZS
        )
        print(f"[ImageNet FULL] Final Zero-shot: Top-1: {top1_final:.2f}%, Top-5: {top5_final:.2f}%")
        if EVAL_OXFORD_PET and pet_val_loader is not None:
            pet_t1_f, pet_t5_f = evaluate_zero_shot(
                backbone, projector, pet_val_loader, pet_zs_weights,
                DEVICE, logit_scale=t_logit_scale, emb_gen_block=emb_gen_block, use_gen=USE_GEN_FOR_ZS
            )
            print(f"[Oxford-Pet] Final Zero-shot: Top-1: {pet_t1_f:.2f}%, Top-5: {pet_t5_f:.2f}%")
        print("\nDistillation training finished.")

    except FileNotFoundError as e:
        print("Error: Dataset directory not found. Check paths.")
        print(e)
        return


if __name__ == '__main__':
    run_distillation()