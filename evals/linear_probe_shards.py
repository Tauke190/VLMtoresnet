import os
import glob
import json
import argparse
import numpy as np
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from linear_probe_extractor import save_split_features  # uses sharding

DATA_ROOT = "/mnt/SSD2/ImageNet1k"
BATCH_SIZE = 256
NUM_WORKERS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

def iter_shards(dir_path: str, split: str):
    for p in sorted(glob.glob(os.path.join(dir_path, f"{split}_*.npz"))):
        data = np.load(p)
        yield data["X"], data["y"]

def l2_normalize(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def build_backbone(checkpoint_path: str | None):
    model = models.resnet50(weights=None)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    return torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()

def extract_phase(backbone, features_dir, shard_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT,"train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT,"validation"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    save_split_features(backbone, train_loader, out_dir=features_dir, split_name="train", shard_size=shard_size)
    save_split_features(backbone, val_loader,   out_dir=features_dir, split_name="val",   shard_size=shard_size)

def train_probe(features_dir, epochs, out_path):
    # Collect class set
    classes = set()
    for X,y in iter_shards(features_dir,"train"):
        classes.update(np.unique(y).tolist())
    classes = np.array(sorted(list(classes)))

    scaler = StandardScaler(with_mean=False)
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                        learning_rate="optimal", max_iter=1, warm_start=True)

    for ep in range(epochs):
        for X,y in iter_shards(features_dir,"train"):
            X = l2_normalize(X)
            scaler.partial_fit(X)
            Xs = scaler.transform(X)
            clf.partial_fit(Xs, y, classes=classes)

        # Validation
        ys_all, yp_all = [], []
        probs_all = []
        for Xv,yv in iter_shards(features_dir,"val"):
            Xv = l2_normalize(Xv)
            Xv = scaler.transform(Xv)
            pv = clf.predict_proba(Xv)
            probs_all.append(pv)
            yp = np.argmax(pv, axis=1)
            ys_all.append(yv)
            yp_all.append(yp)
        ys_all = np.concatenate(ys_all)
        yp_all = np.concatenate(yp_all)
        probs_cat = np.concatenate(probs_all, axis=0)
        top1 = accuracy_score(ys_all, clf.classes_[yp_all])
        # top5
        top5_idx = np.argsort(probs_cat, axis=1)[:,-5:]
        top5_hits = [
            ys_all[i] in clf.classes_[top5_idx[i]]
            for i in range(len(ys_all))
        ]
        top5 = np.mean(top5_hits)
        print(f"Epoch {ep+1}/{epochs}  Top1: {top1*100:.2f}%  Top5: {top5*100:.2f}%")

    joblib.dump({"clf": clf, "scaler": scaler, "classes": classes}, out_path)
    print(f"Saved probe to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--features_dir", type=str, default="features")
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--probe_only", action="store_true")
    parser.add_argument("--out_probe", type=str, default="linear_probe.joblib")
    args = parser.parse_args()

    os.makedirs(args.features_dir, exist_ok=True)

    if not args.probe_only:
        backbone = build_backbone(args.checkpoint)
        extract_phase(backbone, args.features_dir, args.shard_size)
        if args.extract_only:
            return

    train_probe(args.features_dir, args.epochs, args.out_probe)

if __name__ == "__main__":
    main()