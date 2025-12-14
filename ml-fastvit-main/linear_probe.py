import argparse
import os

import clip
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from timm.models import create_model, safe_model_name, load_checkpoint
from CLIP.dataloaders import aircraft as aircraft_dataloader
from CLIP.dataloaders.oxford_pets import OxfordPets
from CLIP.dataloaders.food101 import Food101
import models  # registers custom FastViT models


def load_backbone(args, device):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
    )
    load_checkpoint(model, args.model_checkpoint, use_ema=False)
    model.to(device)
    model.eval()
    print(f"Loaded backbone {safe_model_name(args.model)} from {args.model_checkpoint}")
    return model


def load_projector(projector_ckpt_path, device):
    ckpt = torch.load(projector_ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    w = state["weight"]
    in_dim = w.shape[1]
    out_dim = w.shape[0]

    projector = torch.nn.Linear(in_dim, out_dim)
    projector.load_state_dict(state, strict=True)
    projector.to(device)
    projector.eval()

    print(
        f"Loaded projector from {projector_ckpt_path} "
        f"(in_dim={in_dim}, out_dim={out_dim})"
    )
    return projector


def setup_linearprobe_loaders(dataset_name, dataset_root, batch_size=128, num_workers=4):
    """
    Build train & test loaders using the same CLIP preprocess as in zero-shot eval.
    """
    _, preprocess = clip.load("ViT-L/14", device="cpu", jit=False)

    if dataset_name == "aircraft":
        train_ds = aircraft_dataloader(root=dataset_root, train=True, transform=preprocess)
        test_ds = aircraft_dataloader(root=dataset_root, train=False, transform=preprocess)
        class_names = getattr(train_ds, "categories", None) or getattr(train_ds, "classes", None)
    elif dataset_name == "imagenet":
        from timm.data import create_dataset

        train_ds = create_dataset(
            "",
            root=dataset_root,
            split="train",
            is_training=True,
            transform=preprocess,
        )
        test_ds = create_dataset(
            "",
            root=dataset_root,
            split="validation",
            is_training=False,
            transform=preprocess,
        )
        classes_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
        with open(classes_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
    elif dataset_name == "oxfordpet":
        train_ds = OxfordPets(root=dataset_root, train=True, transform=preprocess)
        test_ds = OxfordPets(root=dataset_root, train=False, transform=preprocess)
        class_names = getattr(train_ds, "categories", None) or getattr(train_ds, "classes", None)
    elif dataset_name == "food101":
        train_ds = Food101(root=dataset_root, train=True, transform=preprocess)
        test_ds = Food101(root=dataset_root, train=False, transform=preprocess)
        class_names = getattr(train_ds, "categories", None) or getattr(train_ds, "classes", None)
    else:
        raise ValueError(f"Unsupported dataset for linear probe: {dataset_name}")

    if class_names is None:
        raise RuntimeError(f"Dataset {dataset_name} has no 'categories' or 'classes' attribute.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # order doesn't matter for logistic regression
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, class_names


def extract_features(loader, backbone, projector, device, use_projector=True):
    """
    Extract features using FastViT (+ optional projector), similar to zeroeval.py.
    """
    all_features = []
    all_labels = []

    model = backbone
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    if projector is not None:
        projector.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features", leave=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # backbone feature extraction
            if hasattr(model, "forward_features"):
                feats = model.forward_features(images)
            else:
                feats = model(images)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4:
                feats = feats.mean(dim=[2, 3])

            feats = feats.float()

            # optional projector (e.g., to CLIP space)
            if use_projector and projector is not None:
                feats = projector(feats)

            # optional normalization (often helps for linear probes)
            feats = F.normalize(feats, dim=-1)

            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return features, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe eval (FastViT features + logistic regression)")
    parser.add_argument("--model", type=str, default="fastvit_sa36", help="Backbone model name")
    parser.add_argument("--num-classes", type=int, default=1000, help="Num classes of backbone head (unused)")
    parser.add_argument("--gp", type=str, default=None, help="Global pool type for timm.create_model")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to backbone checkpoint (.pth.tar)")
    parser.add_argument("--projector-checkpoint", type=str, default=None, help="Path to projector checkpoint (.pth.tar)")
    parser.add_argument("--use-projector", action="store_true", help="Use projector output as features")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (aircraft, imagenet, oxfordpet, food101)")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--C", type=float, default=0.316, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for LogisticRegression")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = setup_linearprobe_loaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"{args.dataset}: {len(class_names)} classes, "
          f"{len(train_loader.dataset)} train images, {len(test_loader.dataset)} test images")

    backbone = load_backbone(args, device)

    projector = None
    if args.projector_checkpoint is not None:
        projector = load_projector(args.projector_checkpoint, device)

    print("Extracting train features...")
    train_features, train_labels = extract_features(
        train_loader,
        backbone,
        projector,
        device,
        use_projector=args.use_projector,
    )

    print("Extracting test features...")
    test_features, test_labels = extract_features(
        test_loader,
        backbone,
        projector,
        device,
        use_projector=args.use_projector,
    )

    print("Fitting logistic regression (linear probe)...")
    clf = LogisticRegression(
        random_state=0,
        C=args.C,
        max_iter=args.max_iter,
        verbose=1,
        n_jobs=-1,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(train_features, train_labels)

    # Top-1 accuracy
    preds = clf.predict(test_features)
    acc1 = (test_labels == preds).astype(float).mean() * 100.0

    # Top-5 accuracy
    probs = clf.predict_proba(test_features)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    acc5 = np.mean([label in top5_row for label, top5_row in zip(test_labels, top5)]) * 100.0

    print(f"\nLinear probe accuracy on {args.dataset}:")
    print(f"  Top-1: {acc1:.3f}%")
    print(f"  Top-5: {acc5:.3f}% ({len(class_names)} classes)")


if __name__ == "__main__":
    main()