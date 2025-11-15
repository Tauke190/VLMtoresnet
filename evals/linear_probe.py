import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, SGDClassifier
import argparse

# IMAGE_NET = '~/data/datasets/imagenet'
IMAGE_NET = '/home/c3-0/datasets/ImageNet'
OXFORD_PET = '~/data/datasets/oxford_pet'

DATASET_PATHS = {
    "imagenet": IMAGE_NET,
    "oxfordpet": OXFORD_PET,
}

parser = argparse.ArgumentParser(description="Linear probe on ResNet50 features")
parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint with backbone_state_dict')
parser.add_argument('--dataset', type=str, choices=DATASET_PATHS.keys(), default="imagenet", help='Dataset to use')
# NEW: classifier + SGD hyperparams
parser.add_argument('--classifier', type=str, choices=['logreg', 'sgd'], default='logreg',
                    help='logreg: scikit-learn LogisticRegression on precomputed features; '
                         'sgd: streaming SGDClassifier with partial_fit')
parser.add_argument('--sgd-epochs', type=int, default=1, help='Number of passes over the training set for SGD')
parser.add_argument('--sgd-alpha', type=float, default=1e-4, help='L2 regularization strength (alpha) for SGD')
args = parser.parse_args()

DATA_ROOT = DATASET_PATHS[args.dataset]

BATCH_SIZE = 32
NUM_WORKERS = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Argument parser for checkpoint only
# parser = argparse.ArgumentParser(description="Linear probe on ResNet50 features")
# parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint with backbone_state_dict')
# args = parser.parse_args()

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

# NEW: streaming SGD train/eval (no feature caching)
def train_sgd_stream(train_dataset, epochs=1, alpha=1e-4):
    clf = SGDClassifier(
        loss="log_loss",  # enables predict_proba
        alpha=alpha,
        penalty="l2",
        learning_rate="optimal",
        fit_intercept=True,
        random_state=0,
    )
    classes = np.arange(len(train_dataset.classes))
    dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    with torch.no_grad():
        for epoch in range(epochs):
            for images, labels in tqdm(dataloader, desc=f"SGD train (epoch {epoch+1}/{epochs})"):
                feats = feature_extractor(images.to(device))
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                clf.partial_fit(feats, labels.numpy(), classes=classes)
    return clf

def evaluate_stream(clf, val_dataset):
    dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    total = 0
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            feats = feature_extractor(images.to(device))
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            probs = clf.predict_proba(feats)
            # Top-1
            top1_idx = np.argmax(probs, axis=1)
            top1_preds = clf.classes_[top1_idx]
            y = labels.numpy()
            top1_correct += np.sum(top1_preds == y)
            # Top-5
            top5_idx = np.argsort(probs, axis=1)[:, -5:]
            top5_classes = clf.classes_[top5_idx]
            top5_correct += sum(y[i] in top5_classes[i] for i in range(len(y)))
            total += len(y)
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5

# Branch by classifier type
if args.classifier == "logreg":
    print("\nExtracting train features...")
    train_features, train_labels = get_features(train, desc="Train features")
    # Upcast once to avoid internal sklearn copy

    print("\nTraining logistic regression classifier...")
    classifier = LogisticRegression(
        random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1
    )
    classifier.fit(train_features, train_labels)

    # Free train cache before eval
    del train_features, train_labels
    import gc; gc.collect()

    print("\nEvaluating (streaming, no val cache)...")
    top1_correct = top5_correct = total = 0
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val (streaming)"):
            feats = feature_extractor(images.to(device)).view(images.size(0), -1).cpu().numpy().astype(np.float64, copy=False)
            probs = classifier.predict_proba(feats)
            y = labels.numpy()
            # Top-1
            top1_idx = np.argmax(probs, axis=1)
            top1_preds = classifier.classes_[top1_idx]
            top1_correct += np.sum(top1_preds == y)
            # Top-5
            top5_idx = np.argsort(probs, axis=1)[:, -5:]
            top5_classes = classifier.classes_[top5_idx]
            top5_correct += sum(y[i] in top5_classes[i] for i in range(len(y)))
            total += len(y)
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
else:
    # Streaming SGD flow (no feature caching)
    print("\nTraining streaming SGD classifier...")
    classifier = train_sgd_stream(train, epochs=args.sgd_epochs, alpha=args.sgd_alpha)
    print("\nEvaluating...")
    top1_acc, top5_acc = evaluate_stream(classifier, val)

print(f"\n{'='*60}")
print(f"ResNet50 Linear Probe Results:")
print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
print(f"{'='*60}")