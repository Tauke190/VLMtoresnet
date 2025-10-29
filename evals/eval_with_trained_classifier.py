# evaluation with trained classifier for imagenet for distilled ResNet-50 model
# python logiteval_distilled_resnet50.py --model-path resnet50_distilled_backbone.pth --classifier-path resnet50_classifier.pth
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm

# --- Configuration ---
VAL_DIR = '~/data/datasets/imagenet/val'  # Path to your ImageNet validation dataset
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
def run_evaluation(model_path):
    print(f"Using device: {DEVICE}")

    # 1. Load the student backbone (distilled)
    print("Loading distilled student backbone...")
    student = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    student.load_state_dict(torch.load(model_path, map_location=DEVICE))
    student.eval()

    # 2. Prepare the validation dataset
    data_config = timm.data.resolve_model_data_config(student)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(val_dataset.classes)

    # 3. Create classifier head
    classifier = nn.Linear(student.num_features, num_classes).to(DEVICE)
    classifier.eval()

    # 4. Run validation
    accuracy = validate_student(student, classifier, val_loader)
    print(f"Validation Accuracy of Distilled Model: {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a distilled ResNet-50 model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the student model file.")
    args = parser.parse_args()

    run_evaluation(args.model_path)