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

# --- Configuration ---
TRAIN_SUBSET_RATIO = 0.15
# For cluster server

# TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
# VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'

TRAIN_DIR = '~/data/datasets/imagenet/train'
VAL_DIR = '~/data/datasets/imagenet/val'
VAL_SUBSET_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

def get_teacher_features(model, images):
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(backbone, images):
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    return pooled_features

def validate_student(backbone, projector, teacher, val_loader):
    backbone.eval()
    projector.eval()
    teacher.eval()
    total_similarity = 0.0
    total_mse = 0.0
    total = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(DEVICE)
            # Student features
            student_features = get_student_features(backbone, images)
            projected_student_features = projector(student_features)
            projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)
            # Teacher features
            teacher_features = teacher.encode_image(images).float()
            teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
            # Cosine similarity (dot product)
            similarity = (projected_student_features * teacher_features).sum(dim=-1)
            total_similarity += similarity.sum().item()
            # MSE loss
            mse = nn.functional.mse_loss(projected_student_features, teacher_features, reduction='sum')
            total_mse += mse.item()
            total += images.size(0)
    avg_similarity = total_similarity / total
    avg_mse = total_mse / total
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")
    print(f"Average MSE Loss: {avg_mse:.6f}")
    return avg_similarity, avg_mse

def load_prompts_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            templates = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(templates)} templates from {filepath}.")
        return templates
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}.")
        return []

def zeroshot_validate_student(backbone, projector, class_names, val_loader, teacher, templates, device=DEVICE):
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

def compute_flops(model, resolution=(3, 224, 224)):
    from thop import profile
    input = torch.randn(1, *resolution).to(DEVICE)
    flops, params = profile(model, inputs=(input,))
    print(f"\n\n***** FLOP TOTAL: {flops / 10 ** 9:.2f} GFLOPs *****")
    print(f"***** Model Parameters: {params:,} *****\n")
    return flops / 10 ** 9, params

def register_hook(module, name, feature_dict):
    def hook_fn(module, input, output):
        # Ensure output is [batch_size, num_tokens, hidden_dim]
        if output.shape[0] != BATCH_SIZE:
            # If output is [num_tokens, batch_size, hidden_dim], transpose
            output = output.permute(1, 0, 2)
        feature_dict[name] = output
    return module.register_forward_hook(hook_fn)

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

    # --- Register hooks for intermediate features ---
    teacher_features_dict = {}
    student_features_dict = {}

    # Example: 6th transformer block for CLIP, 2nd block of layer2 for ResNet
    teacher_handle = register_hook(teacher.visual.transformer.resblocks[5], 'clip_block6', teacher_features_dict)
    student_handle = register_hook(backbone.layer2[1], 'resnet_layer2_1', student_features_dict)

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
        base_train = ImageFolder(root=TRAIN_DIR, transform=train_transform)

        if TRAIN_SUBSET_RATIO is not None and 0.0 < TRAIN_SUBSET_RATIO < 1.0:
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
            train_dataset = Subset(base_train, selected_indices)
            print(f"Using {len(selected_indices)} images (~{TRAIN_SUBSET_RATIO*100:.1f}% per class) from {len(class_to_indices)} classes.")
        else:
            train_dataset = base_train

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        print(f"Loading validation dataset from: {VAL_DIR}")
        full_val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
        val_loader = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Using the full validation dataset with {len(full_val_dataset)} images.")

        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes in the dataset.")

        projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
        distill_loss_fn = nn.MSELoss()
        feature_distill_loss_fn = nn.MSELoss()
        params_to_train = list(backbone.parameters()) + list(projector.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        prompt_file = "prompt/imagenet1k.txt"
        templates = load_prompts_from_file(prompt_file)
        templates = templates[:2]
        class_names = base_train.classes

        print("\nStarting feature distillation...")
        total_start_time = time.time()
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            val_indices = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
            val_subset = Subset(full_val_dataset, val_indices)
            val_loader_subset = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            backbone.train()
            projector.train()
            running_loss = 0.0

            top1, top5 = validate_student(backbone, projector, teacher, val_loader_subset)
            print(f"Validation Accuracy (Logits) after Epoch {epoch+1}: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
            zeroshot_top1, zeroshot_top5 = zeroshot_validate_student(backbone, projector, class_names, val_loader_subset, teacher, templates, DEVICE)
            print(f"Validation Accuracy (Zero-shot) after Epoch {epoch+1}: Top-1: {zeroshot_top1:.2f}%, Top-5: {zeroshot_top5:.2f}%")


            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                teacher_features_dict.clear()
                student_features_dict.clear()

                # Forward pass through teacher and student
                with torch.no_grad():
                    _ = teacher.encode_image(images)
                _ = backbone(images)

                # Get final features
                teacher_features = get_teacher_features(teacher, images).float()
                student_features = get_student_features(backbone, images)
                projected_student_features = projector(student_features)
                teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                projected_student_features = projected_student_features / projected_student_features.norm(dim=-1, keepdim=True)
                loss_distill = distill_loss_fn(projected_student_features, teacher_features)

                # --- Intermediate feature distillation ---
                # CLS token from CLIP block
                teacher_block_feat = teacher_features_dict['clip_block6']  # [B, seq_len, C]
                teacher_cls_token = teacher_block_feat[:, 0]  # CLS token
                teacher_cls_token = teacher_cls_token.float()  # <-- Add this line


                # ResNet intermediate feature
                student_block_feat = student_features_dict['resnet_layer2_1']  # [B, C, H, W]
                student_pooled = nn.functional.adaptive_avg_pool2d(student_block_feat, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]


              
                # Project student pooled feature to match teacher CLS dim if needed
                if student_pooled.shape[1] != teacher_cls_token.shape[1]:
                    if not hasattr(run_distillation, 'student_proj'):
                        run_distillation.student_proj = nn.Linear(student_pooled.shape[1], teacher_cls_token.shape[1]).to(DEVICE)
                    student_pooled_proj = run_distillation.student_proj(student_pooled)
                else:
                    student_pooled_proj = student_pooled


                # print("teacher_block_feat shape:", teacher_block_feat.shape)
                # print("teacher_cls_token shape:", teacher_cls_token.shape)
                # print("student_pooled_proj shape:", student_pooled_proj.shape)

                # Normalize
                teacher_cls_token = teacher_cls_token.float()
                teacher_cls_token = teacher_cls_token / teacher_cls_token.norm(dim=-1, keepdim=True)
                student_pooled_proj = student_pooled_proj / student_pooled_proj.norm(dim=-1, keepdim=True)

                feature_distill_loss = feature_distill_loss_fn(student_pooled_proj, teacher_cls_token)

                total_loss = loss_distill + feature_distill_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

                if i == 999:
                    elapsed_time = time.time() - epoch_start_time
                    avg_time_per_batch = elapsed_time / 1000
                    total_batches = len(train_loader) * NUM_EPOCHS
                    estimated_total_time = avg_time_per_batch * total_batches
                    estimated_hours = int(estimated_total_time // 3600)
                    estimated_minutes = int((estimated_total_time % 3600) // 60)
                    estimated_seconds = int(estimated_total_time % 60)
                    print(f"Estimated total training time: {estimated_hours} hours, {estimated_minutes} minutes, {estimated_seconds} seconds")

                if (i + 1) % 100 == 0:
                    avg_loss_so_far = running_loss / (i + 1)
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.4f}")

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
                    print("Early stopping triggered: training loss has converged.")
                    break

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Time taken for Epoch {epoch+1}: {epoch_time / 60:.2f} minutes")

            if epoch == 0:
                estimated_total_time = epoch_time * NUM_EPOCHS
                estimated_hours = int(estimated_total_time // 3600)
                estimated_minutes = int((estimated_total_time % 3600) // 60)
                estimated_seconds = int(estimated_total_time % 60)
                print(f"Estimated total training time: {estimated_hours} hours, {estimated_minutes} minutes, {estimated_seconds} seconds")

            zeroshot_top1, zeroshot_top5 = zeroshot_validate_student(backbone, projector, class_names, val_loader_subset, teacher, templates, DEVICE)
            print(f"Validation Accuracy (Zero-shot) after Epoch {epoch+1}: Top-1: {zeroshot_top1:.5f}%, Top-5: {zeroshot_top5:.5f}%")

            avg_sim, mse_loss = validate_student(backbone, projector, teacher, val_loader_subset)
            print(f"Validation (Logits) after Epoch {epoch+1}: Average Similarity: {avg_sim:.5f}%, MSE: {mse_loss:.5f}%")
            print("---------------------------------")

            checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': backbone.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved for epoch {epoch+1}.")
            torch.cuda.empty_cache()

        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = int(total_training_time % 60)
        print(f"\nTotal training time: {hours} hours, {minutes} minutes, {seconds} seconds")
        print("---------------------------------")

        print("\nPerforming final validation with the full validation dataset...")
        zeroshot_top1, zeroshot_top5 = zeroshot_validate_student(backbone, projector, class_names, val_loader, teacher, templates, DEVICE)
        print(f"Final Validation Accuracy (Zero-shot): Top-1: {zeroshot_top1:.2f}%, Top-5: {zeroshot_top5:.2f}%")
        print(f"Using the full validation dataset with {len(full_val_dataset)} images for final validation.")

        avg_sim, mse_loss = validate_student(backbone, projector, teacher, val_loader_subset)
        print(f"Final Validation (Logits) after Epoch {epoch+1}: Average Similarity: {avg_sim:.5f}%, MSE: {mse_loss:.5f}%")
        print("\nDistillation training finished.")

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

    # Remove hooks after training
    teacher_handle.remove()
    student_handle.remove()

if __name__ == '__main__':
    run_distillation()