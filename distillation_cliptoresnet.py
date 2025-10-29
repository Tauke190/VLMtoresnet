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
import random  # Add this import at the top of the file

# --- Configuration ---
# !!! IMPORTANT: Update these paths to your Oxford-IIIT Pet dataset directories.
#TRAIN_DIR = '~/datasets/ImageNet2012nonpub/train/'
#VAL_DIR = '~/datasets/ImageNet2012nonpub/val' # Path for the validation set
TRAIN_SUBSET_RATIO = 0.15
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
    """Evaluates the student model on a given dataloader."""
    student_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = student_model.forward_features(images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

class StudentWithProjector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.backbone.global_pool(features)
        return pooled

def zeroshot_validate_student(student_model, projector, class_names, val_loader, teacher, device=DEVICE):
    """
    Performs zero-shot validation using CLIP text embeddings as class prototypes.
    """
    # Prepare class text prompts
    prompts = [f"a photo of a {name}" for name in class_names]
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = teacher.encode_text(text_tokens).float()
        text_features = projector(text_features)  # Project to student feature space
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

    correct = 0
    total = 0
    student_model.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            student_features = student_model.forward_features(images)
            student_features = student_features / student_features.norm(dim=-1, keepdim=True)  # Normalize
            logits = student_features @ text_features.t()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

def run_distillation():
    print(f"Using device: {DEVICE}")

    # 1. --- Setup Models ---
    print("Loading teacher model (CLIP ViT-L/14)...")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading student model (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim
    student_feature_dim = backbone.num_features

    # 2. --- Setup Dataset and DataLoader ---
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

        # --- Now assign num_classes and build projector/student/classifier ---
        num_classes = len(base_train.classes)
        print(f"Found {num_classes} classes in the dataset.")

        projector = nn.Linear(teacher_feature_dim, student_feature_dim).to(DEVICE)
        student = StudentWithProjector(backbone).to(DEVICE)  # Remove projector from student
        classifier = nn.Linear(student_feature_dim, num_classes).to(DEVICE)

        distill_loss_fn = nn.MSELoss()
        classif_loss_fn = nn.CrossEntropyLoss()
        params_to_train = list(student.parameters()) + list(classifier.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        # --- Fast Total ETA Estimation ---
        N = 100
        print(f"\nEstimating total training time using first {N} batches for {NUM_EPOCHS} epochs...")

        batch_times = []
        for i, (images, labels) in enumerate(train_loader):
            start_batch = time.time()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            teacher_features = get_teacher_features(teacher, images)
            projected_student_features = student.forward_features(images)
            projected_teacher_features = projector(teacher_features.float())
            loss_distill = distill_loss_fn(projected_student_features, projected_teacher_features)
            logits = classifier(projected_student_features)
            loss_classif = classif_loss_fn(logits, labels)
            total_loss = loss_distill + loss_classif
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batch_times.append(time.time() - start_batch)
            if i + 1 == N:
                break
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_batches = len(train_loader)
        est_epoch_time = avg_batch_time * total_batches
        est_total_time = est_epoch_time * NUM_EPOCHS
        est_total_str = time.strftime('%H:%M:%S', time.gmtime(est_total_time))
        print(f"Estimated total training time (based on {N} batches): {est_total_str}")

        print("\nStarting distillation...")
        epoch_times = []

        class_names = base_train.classes  # For zero-shot validation

        for epoch in range(NUM_EPOCHS):

            # picks randomized subset for validation
            val_indices = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
            val_subset = Subset(full_val_dataset, val_indices)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            student.train()
            classifier.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                teacher_features = get_teacher_features(teacher, images)
                projected_teacher_features = projector(teacher_features.float())
                student_features = student.forward_features(images)
                loss_distill = distill_loss_fn(student_features, projected_teacher_features)

                logits = classifier(student_features)
                loss_classif = classif_loss_fn(logits, labels)

                total_loss = loss_distill + loss_classif

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                if (i + 1) % 100 == 0:
                    avg_loss_so_far = running_loss / (i + 1)
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.4f}")

            epoch_loss = running_loss / len(train_loader)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")

            val_accuracy = validate_student(student, classifier, val_loader)
            zeroshot_accuracy = zeroshot_validate_student(student, projector, class_names, val_loader, teacher, DEVICE)
            print(f"Validation Accuracy (Logit-based) after Epoch {epoch+1}: {val_accuracy:.2f}%")
            print(f"Validation Accuracy (Zero-shot) after Epoch {epoch+1}: {zeroshot_accuracy:.2f}%")
            print("---------------------------------")

        print("\nDistillation training finished.")
        torch.save(student.state_dict(), 'resnet50_with_projector.pth')
        torch.save(classifier.state_dict(), 'resnet50_distilled_classifier.pth')

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

if __name__ == '__main__':
    run_distillation()