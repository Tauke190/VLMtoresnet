import os
import time
import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from dataloaders.DatasetLoader import DATASET_LOADERS
from models import get_model, MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["stanford_cars", "gtsrb", "food101", "aircraft", "sst2", "country211"],
        choices=list(DATASET_LOADERS.keys()),
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Root directory for datasets"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Cache directory for features"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./resnet50_scalekd_e300.pth",
        help="Checkpoint path for ScaleKD",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--download", action="store_true", help="Download datasets if not present"
    )
    return parser.parse_args()


def get_features(model, extract_fn, dataloader, dataset_name, split, cache_dir, device):
    """Extract and cache features"""
    cache_features_path = os.path.join(
        cache_dir, f"{dataset_name}_{split}_features.npy"
    )
    cache_labels_path = os.path.join(cache_dir, f"{dataset_name}_{split}_labels.npy")

    if os.path.exists(cache_features_path) and os.path.exists(cache_labels_path):
        print(f"  Loading cached {split} features...")
        features = np.load(cache_features_path)
        labels = np.load(cache_labels_path)
        return features, labels

    print(f"  Extracting {split} features...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"  {split}"):
            features = extract_fn(images.to(device))
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()

    np.save(cache_features_path, features)
    np.save(cache_labels_path, labels)
    print(f"  Cached to {cache_features_path}")

    return features, labels


def evaluate_dataset(
    dataset_name,
    model,
    extract_fn,
    transform,
    args,
    cache_dir,
    device,
):
    """Evaluate on a single dataset"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        loader_fn = DATASET_LOADERS[dataset_name]
        train_loader, test_loader = loader_fn(
            data_root=args.data_root,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )

        train_features, train_labels = get_features(
            model, extract_fn, train_loader, dataset_name, "train", cache_dir, device
        )
        test_features, test_labels = get_features(
            model, extract_fn, test_loader, dataset_name, "test", cache_dir, device
        )

        print(f"\n  Train: {train_features.shape}, Test: {test_features.shape}")

        print("  Training logistic regression...")
        classifier = LogisticRegression(
            random_state=0, C=0.316, max_iter=1000, verbose=0, n_jobs=-1
        )
        classifier.fit(train_features, train_labels)

        predictions = classifier.predict(test_features)
        top1_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0

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
        import traceback

        traceback.print_exc()
        return None


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = args.cache_dir or f"./feature_cache_{args.model}"
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 80)
    print(f"Linear Probe Evaluation: {args.model.upper()}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Cache dir: {cache_dir}")
    print(f"Batch size: {args.batch_size}\n")

    model, transform, extract_fn = get_model(
        args.model, device, checkpoint_path=args.checkpoint
    )

    print(f"Model loaded on {device}\n")

    results = {}
    for dataset_name in args.datasets:
        result = evaluate_dataset(
            dataset_name,
            model,
            extract_fn,
            transform,
            args,
            cache_dir,
            device,
        )
        if result:
            results[dataset_name] = result

    print(f"\n\n{'='*80}")
    print(f"{args.model.upper()} - LINEAR PROBING RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Dataset':<20} {'Train':>8} {'Test':>8} {'Top-1':>8} {'Top-5':>8} {'Time(min)':>10}"
    )
    print("-" * 80)

    for dataset_name, res in results.items():
        print(
            f"{dataset_name:<20} {res['train_size']:>8} {res['test_size']:>8} "
            f"{res['top1']:>7.2f}% {res['top5']:>7.2f}% {res['time']:>8.2f}"
        )

    print("=" * 80)

    if results:
        avg_top1 = np.mean([r["top1"] for r in results.values()])
        avg_top5 = np.mean([r["top5"] for r in results.values()])
        avg_time = np.mean([r["time"] for r in results.values()])
        total_time = np.sum([r["time"] for r in results.values()])
        print(
            f"{'Average':<20} {'':<8} {'':<8} {avg_top1:>7.2f}% {avg_top5:>7.2f}% {avg_time:>8.2f}"
        )
        print(f"{'Total Time':<20} {'':<8} {'':<8} {'':<8} {'':<8} {total_time:>8.2f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
