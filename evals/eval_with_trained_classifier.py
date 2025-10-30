import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# --- Configuration ---
VAL_DIR = '~/data/datasets/imagenet/val'
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate_model(backbone, classifier, val_loader):
    backbone.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = backbone.forward_features(images)
            pooled = backbone.global_pool(features)
            outputs = classifier(pooled)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_evaluation(checkpoint_path):
    print(f"Using device: {DEVICE}")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Load backbone
    print("Loading distilled ResNet-50 backbone...")
    backbone = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    backbone.eval()

    # Load classifier
    print("Loading trained classifier...")
    num_classes = checkpoint['classifier_state_dict']['weight'].shape[0]
    classifier = torch.nn.Linear(backbone.num_features, num_classes).to(DEVICE)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.eval()

    # Prepare validation dataset
    data_config = timm.data.resolve_model_data_config(backbone)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Run validation
    accuracy = validate_model(backbone, classifier, val_loader)
    print(f"Validation Accuracy of Distilled Model with Trained Classifier: {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a distilled ResNet-50 model with trained classifier.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint file.")
    args = parser.parse_args()

    run_evaluation(args.checkpoint_path)
