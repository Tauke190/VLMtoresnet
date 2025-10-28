import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm

# --- Configuration ---
TRAIN_DIR = '/home/av354855/data/datasets/imagenet/train'  # Path to your training dataset
VAL_DIR = '/home/av354855/data/datasets/imagenet/val'  # Path to your validation dataset
TRAIN_SUBSET_SIZE = 50000
VAL_SUBSET_SIZE = 50000
BATCH_SIZE = 64
EPOCHS = 10
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
def linear_evaluation():
    print(f"Using device: {DEVICE}")

    # 1. Load the student backbone (distilled)
    print("Loading distilled student backbone...")
    student = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    student.load_state_dict(torch.load('../resnet50_distilled_backbone.pth', map_location=DEVICE))
    student.eval()

    # 2. Prepare the training and validation datasets
    data_config = timm.data.resolve_model_data_config(student)
    transform = timm.data.create_transform(**data_config, is_training=True)

    train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    val_dataset = ImageFolder(root=VAL_DIR, transform=transform)

    train_subset = Subset(train_dataset, range(min(TRAIN_SUBSET_SIZE, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(VAL_SUBSET_SIZE, len(val_dataset))))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
    linear_evaluation()