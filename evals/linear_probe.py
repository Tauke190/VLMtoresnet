import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import argparse

# DATA_ROOT = "/mnt/SSD2/imagenet/"
DATA_ROOT ='/home/c3-0/datasets/ImageNet'

BATCH_SIZE = 256
NUM_WORKERS = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Argument parser for checkpoint only
parser = argparse.ArgumentParser(description="Linear probe on ResNet50 features")
parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint with backbone_state_dict')
args = parser.parse_args()

# Load ResNet50 and remove the final classifier
print("Loading Distilled ResNet50...")

if args.checkpoint:
    model = models.resnet50(weights=None).to(device)
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    print("Checkpoint loaded.")

feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()

# Data transforms
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
print("Loading datasets...")
train = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
val = datasets.ImageFolder(os.path.join(DATA_ROOT, "validation"), transform=transform)
print(f"Train samples: {len(train):,}")
print(f"Val samples: {len(val):,}")

def get_features(dataset, desc="Extracting"):
    all_features = []
    all_labels = []
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            features = feature_extractor(images.to(device))
            features = features.view(features.size(0), -1)  # flatten
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

# Extract features
print("\nExtracting train features...")
train_features, train_labels = get_features(train, desc="Train features")
print(f"Train features shape: {train_features.shape}")
print("\nExtracting val features...")
val_features, val_labels = get_features(val, desc="Val features")
print(f"Val features shape: {val_features.shape}")

# Train logistic regression
print("\nTraining logistic regression classifier...")
classifier = LogisticRegression(
    random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1
)
classifier.fit(train_features, train_labels)
print("\nEvaluating...")
probs = classifier.predict_proba(val_features)
# Top-1 accuracy
top1_preds = np.argmax(probs, axis=1)
top1_acc = np.mean(classifier.classes_[top1_preds] == val_labels) * 100
top5_preds = np.argsort(probs, axis=1)[:, -5:]
top5_acc = (
    np.mean(
        [
            val_labels[i] in classifier.classes_[top5_preds[i]]
            for i in range(len(val_labels))
        ]
    )
    * 100
)
print(f"\n{'='*60}")
print(f"ResNet50 Linear Probe Results:")
print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
print(f"{'='*60}")