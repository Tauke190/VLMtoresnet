# Logit distillation + Supervised finetuning + Contrastive Representational Distillation (CRD)
# Description:
# A script to perform knowledge distillation from a CLIP ViT-L/14 teacher to a
# ResNet-50 student, with validation on a subset of the validation set
# after each epoch. This version is configured for the Oxford-IIIT Pet Dataset.
# CRD: uses CLIP text embeddings (class prototypes) as anchors and student image features as queries.
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
import json
from pathlib import Path

# --- Configuration ---
TRAIN_SUBSET_RATIO = 0.3
# For cluster server

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

# New eval config
EVAL_FULL_VAL_EACH_EPOCH = True
EVAL_OXFORD_PET = True
OXFORD_PET_VAL_DIR = '~/data/datasets/oxford_pet/val'  # ImageFolder layout

# --- CRD configuration ---
USE_CRD = True
CRD_WEIGHT = 0.9  # weight for contrastive distillation loss term relative to other losses

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
        # Fallback to default if empty
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

# CRD loss: student image features as queries, CLIP text features (class prototypes) as keys.
# We compute InfoNCE-style loss across all class anchors (one positive = ground-truth class).
def contrastive_distill_loss(student_features_norm, labels, class_text_features_norm, logit_scale=None):
    # student_features_norm: [B, D], normalized
    # class_text_features_norm: [C, D], normalized
    logits = student_features_norm @ class_text_features_norm.t()  # [B, C]
    if logit_scale is not None:
        logits = logits * logit_scale.exp()
    ce = nn.CrossEntropyLoss()
    return ce(logits, labels)

# Custom prompt templates for Oxford-IIIT Pet
OXFORD_PET_PROMPTS = [
    "a photo of a {}",
    "a photo of a {}, a type of pet",
    "a photo of big {}",
    "a photo of a small {}"
]

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
    teacher_feature_dim = teacher.visual.output_dim
    student_feature_dim = backbone.num_features

    print("Computing FLOPs and parameters for the student model...")
    compute_flops(backbone, resolution=(3, 224, 224))

    train_transform = preprocess
    val_transform = preprocess

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
        base_train = ImageFolder(root=TRAIN_DIR, transform=train_transform)

        # --- Take 15% subset from training set ---
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

        # --- Split 15% subset into train/val (e.g., 80/20 split) ---
        val_ratio_within_subset = 0.20  # Change to 0.10 for 10%
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
        val_indices = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
        val_subset = Subset(full_val_dataset, val_indices)
        val_loader_subset = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes in the dataset.")

        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
        distill_loss_fn = nn.MSELoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        params_to_train = list(backbone.parameters()) + list(projector.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        prompt_file = "prompt/imagenet1k.txt"
        # For ImageNet, align names to ImageFolder's class order
        imagenet_class_names_val = imagenet_aligned_classnames(full_val_dataset, "imagenet_class_index.json")
        print("Prepared ImageNet human-readable class names aligned to validation dataset order.")
        # Also prepare class names aligned to TRAIN dataset order for CRD anchors
        imagenet_class_names_train = imagenet_aligned_classnames(base_train, "imagenet_class_index.json")
        print("Prepared ImageNet human-readable class names aligned to training dataset order.")

        templates = load_prompts_from_file(prompt_file)
        # Optional: learnable/fixed logit scale (temperature). We keep it as a parameter but do not add to optimizer to preserve original behavior.
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07)).to(DEVICE)

        # Precompute text features once (chunked) for ImageNet val
        print("Precomputing ImageNet zero-shot text features (chunked) for validation...")
        text_features_imagenet = precompute_text_features(teacher, imagenet_class_names_val, templates, DEVICE, chunk_size=256)
        torch.cuda.empty_cache()

        # Precompute text features for TRAIN classes (for CRD anchors)
        if USE_CRD:
            print("Precomputing ImageNet text features (chunked) for CRD anchors (train classes)...")
            text_features_train = precompute_text_features(teacher, imagenet_class_names_train, templates, DEVICE, chunk_size=256).to(DEVICE)
            # Ensure normalized
            text_features_train = text_features_train / text_features_train.norm(dim=-1, keepdim=True)
            torch.cuda.empty_cache()
        else:
            text_features_train = None

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

        print("\nStarting logit distillation + CRD...")
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
            running_loss = 0.0

            batch_times = []
            for i, (images, labels) in enumerate(train_loader):
                batch_start_time = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    # Teacher image features for MSE distillation
                    teacher_features = get_teacher_features(teacher, images).float()
                    # Student features and projection to teacher dim
                    student_features = get_student_features(backbone, images)
                    projected_student_features = projector(student_features)

                    # Normalize for cosine similarity-based losses
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                    projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)

                    # Distillation (MSE) in the normalized feature space
                    final_feature_loss = distill_loss_fn(projected_student_features, teacher_features)

                    # CRD: contrastive loss between student projected feats and CLIP text anchors
                    if USE_CRD:
                        loss_crd = contrastive_distill_loss(
                            projected_student_features, labels, text_features_train, logit_scale=logit_scale
                        )
                    else:
                        loss_crd = torch.tensor(0.0, device=DEVICE)

                    # Total loss: keep original terms and add CRD
                    total_loss = 0.1 * final_feature_loss + CRD_WEIGHT * loss_crd

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += total_loss.item()

                # --- Timing and ETA estimation ---
                batch_time = time.time() - batch_start_time
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
                    if USE_CRD:
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.4f} (MSE+CRD)")
                    else:
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.4f}")

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