# evaluation with a pretrained classifier head on top of a distilled ResNet-50 backbone

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm  # For progress bar

# --- Configuration ---
VAL_DIR = '~/data/datasets/imagenet/val'  # Path to your ImageNet validation dataset
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utility Functions ---
def validate_model(model, val_loader):
    """Evaluates accuracy on validation set with a progress bar."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- Main Evaluation ---
def run_evaluation(model_path):
    print(f"Using device: {DEVICE}")

    # 1. Load the distilled ResNet-50 backbone
    print("Loading distilled ResNet-50 backbone...")
    student = timm.create_model('resnet50', pretrained=False).to(DEVICE)
    student.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)  # Allow missing keys
    student.eval()

    # 2. Replace the classifier with a pretrained one
    print("Loading pretrained classifier...")
    pretrained_model = timm.create_model('resnet50', pretrained=True).to(DEVICE)
    student.fc = pretrained_model.fc  # Replace the classifier head with the pretrained one

    # 3. Prepare the validation dataset
    data_config = timm.data.resolve_model_data_config(pretrained_model)  # Use pretrained model config
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Run validation
    accuracy = validate_model(student, val_loader)
    print(f"Validation Accuracy of Distilled Model with Pretrained Classifier: {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a distilled ResNet-50 model with a pretrained classifier.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the student model file.")
    args = parser.parse_args()

    run_evaluation(args.model_path)