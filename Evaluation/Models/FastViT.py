import time
import torch
import timm
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from DatasetLoader import *


def extract_features(model, dataloader, device, normalize=True):
    """Extract features from a dataset."""
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)

            # Handle different output types
            if isinstance(features, (list, tuple)):
                features = features[0]

            # Flatten if needed
            if len(features.shape) > 2:
                features = features.squeeze()

            # Normalize features (critical for fair comparison)
            if normalize:
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()

    return features, labels


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train logistic regression and evaluate."""
    # Train classifier
    print("Training logistic regression...")
    classifier = LogisticRegression(
        random_state=0, C=0.316, max_iter=1000, verbose=0, n_jobs=-1
    )
    classifier.fit(train_features, train_labels)

    # Top-1 accuracy
    predictions = classifier.predict(test_features)
    top1_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0

    # Top-5 accuracy
    probas = classifier.predict_proba(test_features)
    top5_preds = np.argsort(probas, axis=1)[:, -5:]
    top5_accuracy = (
        np.mean([label in top5_preds[i] for i, label in enumerate(test_labels)]) * 100.0
    )

    return {
        "top1": top1_accuracy,
        "top5": top5_accuracy,
        "train_size": len(train_labels),
        "test_size": len(test_labels),
    }


def evaluate_dataset(
    dataset_name, loader_fn, model, transform, data_root, device, skip_download=False
):
    """Evaluate model on a single dataset."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Load data
        train_loader, test_loader = loader_fn(
            data_root=data_root,
            transform=transform,
            batch_size=128,
            num_workers=8,
            download=False if skip_download else True,
        )

        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")

        # Extract features
        print("Extracting training features...")
        train_features, train_labels = extract_features(
            model, train_loader, device, normalize=True
        )

        print("Extracting test features...")
        test_features, test_labels = extract_features(
            model, test_loader, device, normalize=True
        )

        print(
            f"Feature shapes - Train: {train_features.shape}, Test: {test_features.shape}"
        )

        # Train and evaluate
        results = train_and_evaluate(
            train_features, train_labels, test_features, test_labels
        )

        print(f"\nTop-1 Accuracy: {results['top1']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5']:.2f}%")

        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        print(f"Time: {elapsed_minutes:.2f} min")

        results["time"] = elapsed_minutes

        return results

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def main():
    # Configuration
    DATA_ROOT = "./data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "fastvit_sa36"
    PRETRAINED = True

    print("=" * 80)
    print(f"Linear Probe Evaluation: {MODEL_NAME}")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Pretrained: {PRETRAINED}\n")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=0)
    model = model.to(DEVICE)
    model.eval()

    # Get preprocessing transform from timm
    data_config = timm.data.resolve_model_data_config(model)
    base_transform = timm.data.create_transform(**data_config, is_training=False)

    # Wrap transform to handle grayscale images (like FER2013)
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            base_transform,
        ]
    )

    print(f"Model loaded")
    print(f"Input size: {data_config.get('input_size', 'N/A')}\n")

    # Dataset configurations (name, loader_fn, skip_download)
    datasets = [
        ("Stanford Cars", get_stanford_cars_loaders, True),
        ("GTSRB", get_gtsrb_loaders, True),
        ("Food101", get_food101_loaders, True),
        ("FGVC Aircraft", get_aircraft_loaders, True),
        ("SST2", get_sst2_loaders, True),
        # ("FER2013", get_fer2013_loaders, True),  # Already exists
        ("Country211", get_country211_loaders, True),
        # ("UCF101", get_ucf101_loaders, True),  # Already exists
    ]

    # Run evaluation on all datasets
    all_results = {}

    for dataset_name, loader_fn, skip_download in datasets:
        result = evaluate_dataset(
            dataset_name, loader_fn, model, transform, DATA_ROOT, DEVICE, skip_download
        )
        if result:
            all_results[dataset_name] = result

    # Print summary
    print("\n\n" + "=" * 80)
    print(f"{MODEL_NAME.upper()} - LINEAR PROBING RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Dataset':<20} {'Train':>8} {'Test':>8} {'Top-1':>8} {'Top-5':>8} {'Time(min)':>10}"
    )
    print("-" * 80)

    for dataset_name, results in all_results.items():
        print(
            f"{dataset_name:<20} {results['train_size']:>8} {results['test_size']:>8} "
            f"{results['top1']:>7.2f}% {results['top5']:>7.2f}% {results['time']:>8.2f}"
        )

    print("=" * 80)

    # Calculate average
    if all_results:
        avg_top1 = np.mean([r["top1"] for r in all_results.values()])
        avg_top5 = np.mean([r["top5"] for r in all_results.values()])
        avg_time = np.mean([r["time"] for r in all_results.values()])
        total_time = np.sum([r["time"] for r in all_results.values()])
        print(
            f"{'Average':<20} {'':<8} {'':<8} {avg_top1:>7.2f}% {avg_top5:>7.2f}% {avg_time:>8.2f}"
        )
        print(f"{'Total Time':<20} {'':<8} {'':<8} {'':<8} {'':<8} {total_time:>8.2f}")
        print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
