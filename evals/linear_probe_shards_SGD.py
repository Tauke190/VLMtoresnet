import os
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, SGDClassifier
import random

# DATA_ROOT = "/mnt/SSD2/ImageNet1k"
DATA_ROOT = "/home/c3-0/datasets/ImageNet"
BATCH_SIZE = 32
NUM_WORKERS = 8  # increase for faster disk IO
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# Feature extraction (sharded to disk to avoid holding all in RAM)
# ------------------------------------------------------------------
@torch.no_grad()
def extract_and_save_features(dataset, split, feature_extractor, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    shard_idx = 0
    total = 0
    feat_dim = None
    for images, labels in tqdm(loader, desc=f"{split} feature shards"):
        feats = feature_extractor(images.to(device))
        feats = feats.view(feats.size(0), -1).cpu().numpy().astype("float32")
        if feat_dim is None:
            feat_dim = feats.shape[1]
        labels = labels.numpy().astype("int32")
        shard_path = os.path.join(out_dir, f"{split}_{shard_idx:06d}.npz")
        np.savez_compressed(shard_path, X=feats, y=labels)
        shard_idx += 1
        total += feats.shape[0]
    # metadata
    meta = {
        "split": split,
        "count": total,
        "feature_dim": int(feat_dim if feat_dim else -1),
        "num_shards": shard_idx,
    }
    with open(os.path.join(out_dir, f"{split}_meta.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")
    print(f"[{split}] saved {total} samples, dim={feat_dim}, shards={shard_idx}")

def list_shards(out_dir, split):
    return sorted(glob.glob(os.path.join(out_dir, f"{split}_*.npz")))

def filter_good_shards(shard_paths, preview=5):
    good, bad = [], []
    for p in shard_paths:
        if not p.endswith(".npz"):
            continue
        try:
            with np.load(p) as d:
                _ = d["X"].shape; _ = d["y"].shape
        except Exception as e:
            bad.append((p, str(e)))
            continue
        good.append(p)
    if bad:
        print(f"[WARN] Skipping {len(bad)} corrupt shard(s). Showing up to {preview}:")
        for p, e in bad[:preview]:
            print(f"  - {os.path.basename(p)}: {e}")
    return good

def load_features(out_dir, split, use_memmap=False):
    shards = list_shards(out_dir, split)
    if not shards:
        raise FileNotFoundError(f"No feature shards found for split '{split}' in {out_dir}")
    # first pass: sizes + dim
    sizes = []
    feat_dim = None
    for p in shards:
        with np.load(p) as d:
            X = d["X"]
            sizes.append(X.shape[0])
            if feat_dim is None:
                feat_dim = X.shape[1]
    total = sum(sizes)
    if use_memmap:
        # unchanged
        mmap_feat_path = os.path.join(out_dir, f"{split}_features.memmap")
        mmap_lab_path = os.path.join(out_dir, f"{split}_labels.memmap")
        X_all = np.memmap(mmap_feat_path, mode="w+", dtype="float32", shape=(total, feat_dim))
        y_all = np.memmap(mmap_lab_path, mode="w+", dtype="int32", shape=(total,))
        offset = 0
        for p in shards:
            with np.load(p) as d:
                X_all[offset:offset+d["X"].shape[0]] = d["X"]
                y_all[offset:offset+d["y"].shape[0]] = d["y"]
                offset += d["X"].shape[0]
        X_all.flush(); y_all.flush()
        X_all = np.memmap(mmap_feat_path, mode="r", dtype="float32", shape=(total, feat_dim))
        y_all = np.memmap(mmap_lab_path, mode="r", dtype="int32", shape=(total,))
        return X_all, y_all
    else:
        X_all = np.empty((total, feat_dim), dtype="float32")
        y_all = np.empty((total,), dtype="int32")
        offset = 0
        for p, n in zip(shards, sizes):
            with np.load(p) as d:
                X_all[offset:offset+n] = d["X"]
                y_all[offset:offset+n] = d["y"]
            offset += n
        return X_all, y_all

def stream_train_sgd(shard_paths, classes, epochs, shuffle, sgd_params):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=sgd_params["alpha"],
        learning_rate="optimal",
        eta0=sgd_params["lr"],
        random_state=0,
        verbose=1,
    )
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(shard_paths)
        pbar = tqdm(shard_paths, desc=f"SGD epoch {epoch+1}/{epochs}")
        initialized = False
        for p in pbar:
            try:
                with np.load(p) as d:
                    X = d["X"]; y = d["y"]
            except Exception as e:
                print(f"[WARN] Skipping corrupt shard during train: {p} ({e})")
                continue
            if not initialized:
                clf.partial_fit(X, y, classes=classes)
                initialized = True
            else:
                clf.partial_fit(X, y)
        if not initialized:
            raise RuntimeError("No valid shards found to initialize SGDClassifier.")
    return clf

def stream_evaluate(clf, shard_paths):
    top1_correct = 0
    top5_correct = 0
    total = 0
    for p in tqdm(shard_paths, desc="Eval (stream)"):
        try:
            with np.load(p) as d:
                X = d["X"]; y = d["y"]
        except Exception as e:
            print(f"[WARN] Skipping corrupt shard during eval: {p} ({e})")
            continue
        probs = clf.predict_proba(X)
        top1 = np.argmax(probs, axis=1)
        top1_correct += np.sum(clf.classes_[top1] == y)
        top5_idx = np.argsort(probs, axis=1)[:, -5:]
        top5_labels = clf.classes_[top5_idx]
        top5_correct += np.sum((top5_labels == y[:, None]).any(axis=1))
        total += y.shape[0]
    if total == 0:
        raise RuntimeError("No valid validation samples after skipping corrupt shards.")
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    return top1_acc, top5_acc

# ------------------------------------------------------------------
# Original pipeline logic preserved (LogisticRegression identical)
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Linear probe (batch or streaming SGD) on ResNet50 features")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint with backbone_state_dict")
    parser.add_argument("--features_dir", type=str, default="features_lp", help="Directory for feature shards")
    parser.add_argument("--reuse_features", action="store_true", help="Skip extraction if shards exist")
    parser.add_argument("--memmap", action="store_true", help="Assemble features into memmap arrays to reduce RAM")
    parser.add_argument("--no_val", action="store_true", help="Skip validation (only needed for debugging)")
    parser.add_argument("--sgd", action="store_true", help="Use streaming SGDClassifier (no full matrix load)")
    parser.add_argument("--sgd_epochs", type=int, default=1, help="Epochs over shards for SGDClassifier")
    parser.add_argument("--sgd_lr", type=float, default=0.01, help="Base learning rate (eta0 if constant)")
    parser.add_argument("--sgd_alpha", type=float, default=0.0001, help="L2 regularization strength (alpha)")
    parser.add_argument("--sgd_shuffle", action="store_true", help="Shuffle shard order each epoch")
    args = parser.parse_args()

    print(f"Using device: {device}")
    print("Loading Distilled / Pretrained ResNet50...")
    if args.checkpoint:
        model = models.resnet50(pretrained=None).to(device)
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        print("Checkpoint loaded.")
    else:
        model = models.resnet50(pretrained="IMAGENET1K_V2").to(device)
        print("Loaded ImageNet pretrained backbone.")

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "validation"), transform=transform)
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples:   {len(val_ds):,}")

    os.makedirs(args.features_dir, exist_ok=True)

    # Extraction phase
    need_train = not list_shards(args.features_dir, "train")
    need_val = not list_shards(args.features_dir, "val") and not args.no_val
    if args.reuse_features and not (need_train or need_val):
        print("Reusing existing feature shards.")
    else:
        if need_train:
            print("Extracting train features (sharded)...")
            extract_and_save_features(train_ds, "train", feature_extractor, args.features_dir)
        else:
            print("Train feature shards already exist.")
        if not args.no_val:
            if need_val:
                print("Extracting val features (sharded)...")
                extract_and_save_features(val_ds, "val", feature_extractor, args.features_dir)
            else:
                print("Val feature shards already exist.")

    train_shards = list_shards(args.features_dir, "train")
    val_shards = list_shards(args.features_dir, "val") if not args.no_val else []

    # Filter out corrupt shards up front
    train_shards = filter_good_shards(train_shards)
    if not train_shards:
        raise RuntimeError("No valid training shards found.")
    if not args.no_val:
        val_shards = filter_good_shards(val_shards)
        if not val_shards:
            print("[WARN] No valid validation shards found; skipping eval.")
            args.no_val = True
    if args.sgd:
        print("\nStreaming SGDClassifier training (no full matrix load)...")
        classes = np.arange(len(train_ds.classes), dtype="int32")
        clf = stream_train_sgd(
            train_shards,
            classes,
            epochs=args.sgd_epochs,
            shuffle=args.sgd_shuffle,
            sgd_params={"lr": args.sgd_lr, "alpha": args.sgd_alpha},
        )
    else:
        print("Loading train feature shards into array (batch training)...")
        train_features, train_labels = load_features(args.features_dir, "train", use_memmap=args.memmap)
        print(f"Train features shape: {train_features.shape}")
        if not args.no_val:
            print("Loading val feature shards into array...")
            val_features, val_labels = load_features(args.features_dir, "val", use_memmap=args.memmap)
            print(f"Val features shape:   {val_features.shape}")
        print("\nTraining batch LogisticRegression classifier...")
        clf = LogisticRegression(
            random_state=0,
            C=0.316,
            max_iter=1000,
            verbose=1,
            n_jobs=-1,
        )
        clf.fit(train_features, train_labels)

    if args.no_val:
        print("Validation skipped (--no_val).")
        return

    print("\nEvaluating...")
    if args.sgd:
        top1_acc, top5_acc = stream_evaluate(clf, val_shards)
    else:
        probs = clf.predict_proba(val_features)
        top1_preds = np.argmax(probs, axis=1)
        top1_acc = np.mean(clf.classes_[top1_preds] == val_labels) * 100
        top5_idx = np.argsort(probs, axis=1)[:, -5:]
        top5_hits = (clf.classes_[top5_idx] == val_labels[:, None]).any(axis=1)
        top5_acc = np.mean(top5_hits) * 100

    print(f"\n{'='*60}")
    mode = "SGD (streaming)" if args.sgd else "LogisticRegression (batch)"
    print(f"ResNet50 Linear Probe Results [{mode}]:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()