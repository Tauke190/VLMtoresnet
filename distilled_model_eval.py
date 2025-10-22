import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm

# --- Configuration ---
VAL_DIR = '~/data/datasets/oxford_pet/val'  # Path to your validation dataset
VAL_SUBSET_SIZE = 10000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utility Functions ---
def get_student_features(model, images):
    """Extract global average pooled features from the student (ResNet-50)."""
    feature_map = model.forward_features(images)
    pooled_features = model.global_pool(feature_map)
    return pooled_features

def validate_student(student_model, classifier, val_loader):
    """Evaluates accuracy on validation set."""
    student_model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = get_student_features(student_model, images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- Main Evaluation ---
def run_evaluation():
    print(f"Using device: {DEVICE}")

    # 1. Load the student backbone (distilled)
    print("Loading distilled student backbone...")
    student = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    student.load_state_dict(torch.load('resnet50_distilled_backbone.pth', map_location=DEVICE))
    student.eval()

    # 2. Prepare the validation dataset
    data_config = timm.data.resolve_model_data_config(student)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    val_subset = Subset(val_dataset, range(min(VAL_SUBSET_SIZE, len(val_dataset))))
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(val_dataset.classes)

    # 3. Load classifier head
    classifier = nn.Linear(student.num_features, num_classes).to(DEVICE)
    classifier.load_state_dict(torch.load('resnet50_distilled_classifier.pth', map_location=DEVICE))
    classifier.eval()

    # 4. Run validation
    accuracy = validate_student(student, classifier, val_loader)
    print(f"Validation Accuracy of Distilled Model: {accuracy:.2f}%")

if __name__ == "__main__":
    run_evaluation()

