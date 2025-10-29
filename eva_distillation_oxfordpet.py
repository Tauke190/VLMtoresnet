# knowledge_distillation.py
#
# Description:
# A script to perform knowledge distillation from an EVA-02 teacher to a
# ResNet-50 student, with validation on a subset of the validation set
# after each epoch. This version is configured for the Oxford-IIIT Pet Dataset.
#
# Dependencies:
# pip install torch torchvision timm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
import clip
import random
import time

# --- Configuration ---
# !!! IMPORTANT: Update these paths to your Oxford-IIIT Pet dataset directories.
TRAIN_DIR = '~/data/datasets/oxford_pet/train'
VAL_DIR = '~/data/datasets/oxford_pet/val' # Path for the validation set

# Only for code development server
# TRAIN_DIR = '~/data/datasets/oxford_pet/train'
# VAL_DIR = '~/data/datasets/oxford_pet/val' 
VAL_SUBSET_SIZE = 1000 # Number of images to use for validation each epoch

BATCH_SIZE = 16  # Adjust based on your GPU memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5 # Increased for demonstration to see progress
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_teacher_features(model, images):
    """Extracts features from the teacher model."""
    with torch.no_grad():
        features = model.forward_features(images)[:, 0]  # Use CLS token
    return features

def get_student_features(model, images):
    """
    Extracts features from the student model's backbone and applies
    global average pooling to get a feature vector.
    """
    # Get the feature map from the ResNet backbone (e.g., shape: [N, 2048, 14, 14])
    feature_map = model.forward_features(images)
    # Apply global average pooling to collapse spatial dimensions (-> [N, 2048])
    # This is necessary to match the teacher's feature vector shape.
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
            features = get_student_features(student_model, images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def compute_flops(model, resolution):
    # Dummy implementation for FLOPs computation
    # Replace with actual FLOPs computation if needed
    print(f"Computing FLOPs for model: {model.__class__.__name__}, Resolution: {resolution}")

def load_prompts_from_file(file_path):
    # Dummy implementation for loading prompts
    # Replace with actual prompt loading logic if needed
    return ["a photo of a {}.", "an image of a {}.", "a snapshot of a {}."]

class StudentWithProjector(nn.Module):
    def __init__(self, backbone):
        super(StudentWithProjector, self).__init__()
        self.backbone = backbone

    def forward_features(self, x):
        return self.backbone(x)

def zeroshot_validate_student(student, projector, class_names, val_loader, teacher, templates, device):
    """Perform zero-shot validation using CLIP-like prompting."""
    student.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Get student features
            student_features = student.forward_features(images)

            # Zero-shot classification head (logits)
            logits = projector(student_features.float())

            # Compute top-k accuracy
            _, predicted_top1 = torch.max(logits, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()

            # For top-5 accuracy, we need to get the top 5 predictions
            _, predicted_top5 = torch.topk(logits, 5, dim=1)
            correct_top5 += (predicted_top5 == labels.view(-1, 1).expand_as(predicted_top5)).sum().item()

            total += labels.size(0)

    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total
    return accuracy_top1, accuracy_top5

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
        classifier = nn.Linear(student_feature_dim, num_classes).to(DEVICE)

        distill_loss_fn = nn.MSELoss()
        params_to_train = list(student.parameters()) + list(classifier.parameters())
        optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

        # --- Load prompts ---
        prompt_file = "prompt/imagenet1k.txt"
        templates = load_prompts_from_file(prompt_file)
        class_names = base_train.classes

        print("\nStarting distillation...")

        # Start timer for total training time
        total_start_time = time.time()

        # Estimate time for the first epoch
        estimated_epoch_time = None

        for epoch in range(NUM_EPOCHS):

            # picks randomized subset for validation
            val_indices = random.sample(range(len(full_val_dataset)), min(VAL_SUBSET_SIZE, len(full_val_dataset)))
            val_subset = Subset(full_val_dataset, val_indices)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            student.train()
            classifier.train()
            running_loss = 0.0

            # Measure time for the first 100 batches
            if epoch == 0:  # Only estimate time during the first epoch
                start_time = time.time()
                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    teacher_features = get_teacher_features(teacher, images)
                    projected_teacher_features = projector(teacher_features.float())
                    student_features = student.forward_features(images)
                    loss_distill = distill_loss_fn(student_features, projected_teacher_features)

                    logits = classifier(student_features)
                    loss_classif = nn.CrossEntropyLoss()(logits, labels)

                    total_loss = loss_distill + loss_classif

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item()

                    # Estimate training time after 100 batches
                    if i == 99:
                        elapsed_time = time.time() - start_time
                        total_batches = len(train_loader)
                        estimated_epoch_time = (elapsed_time / 100) * total_batches
                        estimated_total_time = estimated_epoch_time * NUM_EPOCHS
                        print(f"Estimated training time per epoch: {estimated_epoch_time / 60:.2f} minutes")
                        print(f"Estimated total training time: {estimated_total_time / 3600:.2f} hours")
                        break

            # Continue training after time estimation
            for i, (images, labels) in enumerate(train_loader):
                if epoch == 0 and i < 100:  # Skip the first 100 batches (already processed)
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                teacher_features = get_teacher_features(teacher, images)
                projected_teacher_features = projector(teacher_features.float())
                student_features = student.forward_features(images)
                loss_distill = distill_loss_fn(student_features, projected_teacher_features)

                logits = classifier(student_features)
                loss_classif = nn.CrossEntropyLoss()(logits, labels)

                total_loss = loss_distill + loss_classif

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"\n--- End of Epoch {epoch+1} ---")
            print(f"Average Training Loss: {epoch_loss:.4f}")

            zeroshot_top1, zeroshot_top5 = zeroshot_validate_student(student, projector, class_names, val_loader, teacher, templates, DEVICE)
            print(f"Validation Accuracy (Zero-shot) after Epoch {epoch+1}: Top-1: {zeroshot_top1:.2f}%, Top-5: {zeroshot_top5:.2f}%")
            print("---------------------------------")

        # Final validation on the entire validation dataset
        print("\nPerforming final validation on the entire validation dataset...")

        # Create a DataLoader for the full validation dataset
        full_val_loader = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        # Logit-based validation
        print("\nLogit-based validation:")
        final_logit_top1, final_logit_top5 = validate_student(student, classifier, full_val_loader)
        print(f"Final Validation Accuracy (Logit-based): Top-1: {final_logit_top1:.2f}%, Top-5: {final_logit_top5:.2f}%")

        # Zero-shot validation
        print("\nZero-shot validation:")
        final_zeroshot_top1, final_zeroshot_top5 = zeroshot_validate_student(student, projector, class_names, full_val_loader, teacher, templates, DEVICE)
        print(f"Final Validation Accuracy (Zero-shot): Top-1: {final_zeroshot_top1:.2f}%, Top-5: {final_zeroshot_top5:.2f}%")

        print("\nDistillation training finished.")
        torch.save(student.state_dict(), 'resnet50_with_projector.pth')
        torch.save(backbone.state_dict(), 'resnet50_distilled_with_logit_distillation.pth')

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

if __name__ == '__main__':
    run_distillation()



