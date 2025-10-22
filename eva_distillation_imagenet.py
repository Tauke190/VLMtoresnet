
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

# --- Configuration ---
# !!! IMPORTANT: Update these paths to your Oxford-IIIT Pet dataset directories.
TRAIN_DIR = '~/data/datasets/oxford_pet/train'
VAL_DIR = '~/data/datasets/oxford_pet/val' # Path for the validation set
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

def run_distillation():
    """
    Main function to set up models, data, and run the distillation training loop.
    """
    print(f"Using device: {DEVICE}")

    # 1. --- Setup Models ---
    print("Loading teacher model (EVA-02)...")
    teacher_model_name = 'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k'
    teacher = timm.create_model(teacher_model_name, pretrained=True, num_classes=0).to(DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading student model (ResNet-50)...")
    # Load student backbone without a classifier head
    student = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)

    teacher_feature_dim = teacher.embed_dim
    student_feature_dim = student.num_features

    # 2. --- Setup Dataset and DataLoader ---
    data_config = timm.data.resolve_model_data_config(teacher)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    try:
        print(f"Loading training dataset from: {TRAIN_DIR}")
        train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        print(f"Loading validation dataset from: {VAL_DIR}")
        full_val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
        # Create a smaller subset for faster validation
        val_subset = Subset(full_val_dataset, range(min(VAL_SUBSET_SIZE, len(full_val_dataset))))
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Using a validation subset of {len(val_subset)} images.")
    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. Please check your paths.")
        print(e)
        return

    # Determine number of classes from the dataset
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes in the dataset.")

    # Create a projection layer and a new classifier head for the student
    projection = nn.Linear(teacher_feature_dim, student_feature_dim).to(DEVICE)
    classifier = nn.Linear(student_feature_dim, num_classes).to(DEVICE)

    # 3. --- Setup Loss and Optimizer ---
    distill_loss_fn = nn.MSELoss()
    classif_loss_fn = nn.CrossEntropyLoss()
    params_to_train = list(student.parameters()) + list(projection.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params_to_train, lr=LEARNING_RATE)

    # 4. --- Distillation Training Loop ---
    print("\nStarting distillation...")
    for epoch in range(NUM_EPOCHS):
        student.train()
        classifier.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            teacher_features = get_teacher_features(teacher, images)
            student_features = get_student_features(student, images)

            # --- Calculate Losses ---
            # 1. Distillation Loss
            projected_teacher_features = projection(teacher_features)
            loss_distill = distill_loss_fn(student_features, projected_teacher_features)

            # 2. Classification Loss
            logits = classifier(student_features)
            loss_classif = classif_loss_fn(logits, labels)

            # Combine losses (equal weighting for simplicity)
            total_loss = loss_distill + loss_classif

            # Backpropagation
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

        # --- Validation Step ---
        val_accuracy = validate_student(student, classifier, val_loader)
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy:.2f}%")
        print("---------------------------------")


    print("\nDistillation training finished.")
    # Save the student backbone and the trained classifier
    torch.save(student.state_dict(), 'resnet50_distilled_backbone.pth')
    torch.save(classifier.state_dict(), 'resnet50_distilled_classifier.pth')

if __name__ == '__main__':
    run_distillation()



