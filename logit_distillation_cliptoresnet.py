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
import time  # Import time for measuring training duration
import random  # Add this import at the top of the file
import numpy as np  # Add this import for parameter calculation

# --- Configuration ---
# !!! IMPORTANT: Update these paths to your Oxford-IIIT Pet dataset directories.
#TRAIN_DIR = '~/datasets/ImageNet2012nonpub/train/'
#VAL_DIR = '~/datasets/ImageNet2012nonpub/val' # Path for the validation set
TRAIN_SUBSET_RATIO = 1
# Only for code development server
# TRAIN_DIR = '/datasets/ImageNet2012nonpub/train'
TRAIN_DIR = '~/data/datasets/imagenet/train'
VAL_DIR = '~/data/datasets/imagenet/val'
VAL_SUBSET_SIZE = 5000 # Number of images to use for validation each epoch
BATCH_SIZE = 16  # Adjust based on your GPU memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_teacher_features(model, images):
    """Extracts features from the CLIP teacher model."""
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(model, images):
    """
    Extracts features from the student model's backbone and applies
    global average pooling to get a feature vector.
    """
    feature_map = model.forward_features(images)
    pooled_features = model.global_pool(feature_map)
    return pooled_features

def validate_student(student_model, classifier, val_loader):
    """Evaluates the student model on a given dataloader and calculates top-1 and top-5 accuracy."""
    student_model.eval()
    classifier.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = student_model.forward_features(images)
            outputs = classifier(features)
            _, top5_preds = outputs.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()  # Top-1 accuracy
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()  # Top-5 accuracy
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy

class StudentWithProjector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.backbone.global_pool(features)
        return pooled

def load_prompts_from_file(filepath):
    """
    Loads all prompt templates from a text file.
    Each line in the file represents a template.
    """
    try:
        with open(filepath, 'r') as f:
            templates = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(templates)} templates from {filepath}.")
        return templates
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}.")
        return []

def zeroshot_validate_student(student_model, projector, class_names, val_loader, teacher, templates, device=DEVICE):
    """
    Performs zero-shot validation using CLIP text embeddings as class prototypes.
    Considers all prompt templates for each class and calculates top-1 and top-5 accuracy.
    """
    # Prepare class text prompts using all templates
    prompts = [template.format(name) for name in class_names for template in templates]
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = teacher.encode_text(text_tokens).float()
        text_features = projector(text_features)  # Project to student feature space
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

    # Average the text features for each class
    num_templates = len(templates)
    num_classes = len(class_names)
    text_features = text_features.view(num_classes, num_templates, -1).mean(dim=1)

    top1_correct = 0
    top5_correct = 0
    total = 0
    student_model.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            student_features = student_model.forward_features(images)
            student_features = student_features / student_features.norm(dim=-1, keepdim=True)  # Normalize
            logits = student_features @ text_features.t()
            _, top5_preds = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()  # Top-1 accuracy
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()  # Top-5 accuracy
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy

def compute_flops(model, resolution=(3, 224, 224)):
    """
    Computes the FLOPs and parameters of the given model using thop.
    """
    from thop import profile
    input = torch.randn(1, *resolution).to(DEVICE)
    flops, params = profile(model, inputs=(input,))
    print(f"\n\n***** FLOP TOTAL: {flops / 10 ** 9:.2f} GFLOPs *****")
    print(f"***** Model Parameters: {params:,} *****\n")
    return flops / 10 ** 9, params

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

    # Compute FLOPs and parameters for the student model
    print("Computing FLOPs and parameters for the student model...")
    compute_flops(backbone, resolution=(3, 224, 224))

    # --- Setup Dataset and DataLoader ---
    train_transform = preprocess
    val_transform = preprocess

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
        val_subset = Subset(full_val_dataset, range(min(VAL_SUBSET_SIZE, len(full_val_dataset))))
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Using a validation subset of {len(val_subset)} images.")

        # --- Now assign num_classes and build projector/student ---
        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes in the dataset.")

        projector = nn.Linear(teacher_feature_dim, student_feature_dim).to(DEVICE)
        student = StudentWithProjector(backbone).to(DEVICE)

        distill_loss_fn = nn.MSELoss()
        params_to_train = list(student.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        # --- Load prompts ---
        prompt_file = "prompt/imagenet1k.txt"
        templates = load_prompts_from_file(prompt_file)
        class_names = base_train.classes

        print("\nStarting distillation...")
        for epoch in range(NUM_EPOCHS):

            # picks randomized subset for validation
            val_indices = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
            val_subset = Subset(full_val_dataset, val_indices)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            student.train()
            running_loss = 0.0

            # Measure time for the first 100 batches
            start_time = time.time()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                teacher_features = get_teacher_features(teacher, images)
                projected_teacher_features = projector(teacher_features.float())
                student_features = student.forward_features(images)
                loss_distill = distill_loss_fn(student_features, projected_teacher_features)

                optimizer.zero_grad()
                loss_distill.backward()
                optimizer.step()

                running_loss += loss_distill.item()

                # Estimate training time after 100 batches
                if i == 99:
                    elapsed_time = time.time() - start_time
                    total_batches = len(train_loader)
                    estimated_time = (elapsed_time / 100) * total_batches
                    print(f"Estimated training time per epoch: {estimated_time / 60:.2f} minutes")
                    break

            # Continue training after time estimation
            for i, (images, labels) in enumerate(train_loader):
                if i < 100:  # Skip the first 100 batches (already processed)
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                teacher_features = get_teacher_features(teacher, images)
                projected_teacher_features = projector(teacher_features.float())
                student_features = student.forward_features(images)
                loss_distill = distill_loss_fn(student_features, projected_teacher_features)

                optimizer.zero_grad()
                loss_distill.backward()
                optimizer.step()

                running_loss += loss_distill.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")

            zeroshot_top1, zeroshot_top5 = zeroshot_validate_student(student, projector, class_names, val_loader, teacher, templates, DEVICE)
            print(f"Validation Accuracy (Zero-shot) after Epoch {epoch+1}: Top-1: {zeroshot_top1:.2f}%, Top-5: {zeroshot_top5:.2f}%")
            print("---------------------------------")

        print("\nDistillation training finished.")
        torch.save(student.state_dict(), 'resnet50_with_projector.pth')
        torch.save(backbone.state_dict(), 'resnet50_distilled_with_logit_distillation.pth')

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

if __name__ == '__main__':
    run_distillation()