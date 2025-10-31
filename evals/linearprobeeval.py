# linear probe evaluation script for distilled ResNet-50 model

import argparse
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utility Functions ---
def extract_features(model, dataloader):
    """Extract features from the model for all images in the dataloader."""
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            features = get_student_features(model, images)
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)

def get_student_features(model, images):
    """Extract global average pooled features from the student (ResNet-50)."""
    feature_map = model.forward_features(images)
    pooled_features = model.global_pool(feature_map)  # Shape: [batch_size, 2048]
    return pooled_features

# --- Linear Evaluation ---
def linear_evaluation(model_path, dataset):
    print(f"Using device: {DEVICE}")

    # 1. Load the student backbone (distilled)
    print("Loading distilled student backbone...")
    student = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()

    # 2. Prepare the training and validation datasets
    if dataset == "imagenet":
        train_dir = '/home/av354855/data/datasets/imagenet/train'
        val_dir = '/home/av354855/data/datasets/imagenet/val'
    elif dataset == "oxfordpet":
        train_dir = '/home/av354855/data/datasets/oxford_pet/train'
        val_dir = '/home/av354855/data/datasets/oxford_pet/val'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    print(f"Using dataset: {dataset}")
    data_config = timm.data.resolve_model_data_config(student)
    transform = timm.data.create_transform(**data_config, is_training=True)

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(student, train_loader)

    print("Extracting validation features...")
    val_features, val_labels = extract_features(student, val_loader)

    # 4. Define the linear classifier
    print("Training linear classifier...")
    num_classes = len(train_dataset.classes)
    classifier = nn.Linear(2048, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. Train the linear classifier
    for epoch in range(EPOCHS):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(train_features), BATCH_SIZE):
            inputs = train_features[i:i + BATCH_SIZE].to(DEVICE)
            labels = train_labels[i:i + BATCH_SIZE].to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

    # 6. Evaluate the linear classifier
    print("Evaluating linear classifier...")
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(val_features), BATCH_SIZE):
            inputs = val_features[i:i + BATCH_SIZE].to(DEVICE)
            labels = val_labels[i:i + BATCH_SIZE].to(DEVICE)

            outputs = classifier(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = 100. * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probe Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--dataset", type=str, required=True, choices=["imagenet", "oxfordpet"], help="Dataset to use for evaluation")
    args = parser.parse_args()

    linear_evaluation(args.checkpoint, args.dataset)