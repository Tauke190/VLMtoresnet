import os
import time
import torch
import timm
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from DatasetLoader import *

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
cache_dir = "./feature_cache"
os.makedirs(cache_dir, exist_ok=True)

# Load ConvMixer 768/32
print("Loading ConvMixer 768/32...")
model = timm.create_model(
    "convmixer_768_32", pretrained=True, num_classes=0
)  # num_classes=0 for feature extraction
model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

# Standard transform for ConvMixer (224x224)
# Handles both RGB and grayscale images (converts grayscale to RGB)
transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda x: x.convert("RGB") if x.mode != "RGB" else x
        ),  # Convert grayscale to RGB
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_features(model, dataloader, dataset_name, split):
    """Extract and cache features."""
    cache_features_path = os.path.join(
        cache_dir, f"{dataset_name}_{split}_features.npy"
    )
    cache_labels_path = os.path.join(cache_dir, f"{dataset_name}_{split}_labels.npy")

    # Check cache
    if os.path.exists(cache_features_path) and os.path.exists(cache_labels_path):
        print(f"  Loading cached {split} features...")
        features = np.load(cache_features_path)
        labels = np.load(cache_labels_path)
        return features, labels

    # Extract features
    print(f"  Extracting {split} features...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"  {split}"):
            features = model(images.to(device))
            if len(features.shape) > 2:
                features = features.squeeze()
            features = features / (
                features.norm(dim=-1, keepdim=True) + 1e-8
            )  # Normalize
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()

    # Save cache
    np.save(cache_features_path, features)
    np.save(cache_labels_path, labels)
    print(f"  Cached to {cache_features_path}")

    return features, labels


def evaluate_dataset(dataset_name, loader_fn, skip_download=False):
    """Evaluate on a single dataset."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Get data loaders
        train_loader, test_loader = loader_fn(
            data_root=data_root,
            transform=transform,
            batch_size=128,
            num_workers=8,
            download=False if skip_download else True,
        )

        # Extract features
        train_features, train_labels = get_features(
            model, train_loader, dataset_name, "train"
        )
        test_features, test_labels = get_features(
            model, test_loader, dataset_name, "test"
        )

        print(f"\n  Train: {train_features.shape}, Test: {test_features.shape}")

        # Train linear probe
        print("  Training logistic regression...")
        classifier = LogisticRegression(
            random_state=0, C=0.316, max_iter=1000, verbose=0, n_jobs=-1
        )
        classifier.fit(train_features, train_labels)

        # Evaluate
        predictions = classifier.predict(test_features)
        top1_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0

        # Top-5 accuracy
        probas = classifier.predict_proba(test_features)
        top5_preds = np.argsort(probas, axis=1)[:, -5:]
        top5_accuracy = (
            np.mean([label in top5_preds[i] for i, label in enumerate(test_labels)])
            * 100.0
        )

        print(f"\n  Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"  Top-5 Accuracy: {top5_accuracy:.2f}%")

        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        print(f"  Time: {elapsed_minutes:.2f} min")

        return {
            "top1": top1_accuracy,
            "top5": top5_accuracy,
            "train_size": len(train_labels),
            "test_size": len(test_labels),
            "time": elapsed_minutes,
        }

    except Exception as e:
        print(f"\n  Error: {str(e)}")
        return None


# Dataset configurations (name, loader_fn, skip_download)
datasets_config = [
    ("Stanford Cars", get_stanford_cars_loaders, True), # True: Skip_Download
    ("GTSRB", get_gtsrb_loaders, True),
    ("Food101", get_food101_loaders, True),
    ("FGVC Aircraft", get_aircraft_loaders, True),
    ("SST2", get_sst2_loaders, True),
    # ("FER2013", get_fer2013_loaders, True),  # Already exists
    ("Country211", get_country211_loaders, True),
    # ("UCF101", get_ucf101_loaders, True),  # Already exists
]

# Run evaluations
results = {}
for dataset_name, loader_fn, skip_download in datasets_config:
    result = evaluate_dataset(dataset_name, loader_fn, skip_download)
    if result:
        results[dataset_name] = result

# Print summary
print(f"\n\n{'='*70}")
print("CONVMIXER 768/32 - LINEAR PROBING RESULTS SUMMARY")
print(f"{'='*70}")
print(
    f"{'Dataset':<20} {'Train':>8} {'Test':>8} {'Top-1':>8} {'Top-5':>8} {'Time(min)':>10}"
)
print(f"{'-'*70}")
for name, res in results.items():
    print(
        f"{name:<20} {res['train_size']:>8} {res['test_size']:>8} {res['top1']:>7.2f}% {res['top5']:>7.2f}% {res['time']:>8.2f}"
    )
print(f"{'='*70}")

# Calculate average
if results:
    avg_top1 = np.mean([r["top1"] for r in results.values()])
    avg_top5 = np.mean([r["top5"] for r in results.values()])
    avg_time = np.mean([r["time"] for r in results.values()])
    total_time = np.sum([r["time"] for r in results.values()])
    print(
        f"{'Average':<20} {'':<8} {'':<8} {avg_top1:>7.2f}% {avg_top5:>7.2f}% {avg_time:>8.2f}"
    )
    print(f"{'Total Time':<20} {'':<8} {'':<8} {'':<8} {'':<8} {total_time:>8.2f}")
    print(f"{'='*70}")
