import os
import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import timm
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_TRAIN_DIR ='/home/c3-0/datasets/ImageNet/train'
IMAGENET_VAL_DIR   = '/home/c3-0/datasets/ImageNet/validation'
# IMAGENET_TRAIN_DIR = os.path.expanduser('~/data/datasets/imagenet/train')
# IMAGENET_VAL_DIR   = os.path.expanduser('~/data/datasets/imagenet/val')



OXFORD_PET_TRAIN_DIR = os.path.expanduser('~/data/datasets/oxford_pet/train')
OXFORD_PET_VAL_DIR   = os.path.expanduser('~/data/datasets/oxford_pet/val')

def preprocess():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

def load_backbone_and_projector(checkpoint_path: str):
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    backbone_sd = ckpt.get('backbone_state_dict', {})
    projector_sd = ckpt.get('projector_state_dict', {})

    # Create backbone exactly like snippet
    backbone = timm.create_model('resnet50', pretrained=False, num_classes=0).to(device)
    # Clean possible 'module.' prefixes
    backbone_sd = { (k[7:] if k.startswith('module.') else k): v for k, v in backbone_sd.items() }
    missing = backbone.load_state_dict(backbone_sd, strict=False)
    print(f"Backbone loaded. Missing={len(missing.missing_keys)} Unexpected={len(missing.unexpected_keys)}")

    # Projector: assume simple Linear
    if 'weight' in projector_sd and projector_sd['weight'].ndim == 2:
        out_dim, in_dim = projector_sd['weight'].shape
        projector = nn.Linear(in_dim, out_dim, bias=('bias' in projector_sd)).to(device)
        projector.load_state_dict(projector_sd, strict=False)
    else:
        print("Warning: projector_state_dict not recognized, using identity.")
        projector = nn.Identity().to(device)

    backbone.eval()
    projector.eval()
    return backbone, projector

@torch.no_grad()
def encode_images(backbone, projector, images: torch.Tensor):
    if hasattr(backbone, "forward_features") and hasattr(backbone, "global_pool"):
        fmap = backbone.forward_features(images)
        pooled = backbone.global_pool(fmap).flatten(1)
    else:
        out = backbone(images)
        pooled = out if out.ndim == 2 else torch.flatten(out, 1)
    feats = projector(pooled)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

def load_datasets(name, tfm):
    if name == "imagenet":
        assert os.path.isdir(IMAGENET_TRAIN_DIR) and os.path.isdir(IMAGENET_VAL_DIR), "ImageNet paths invalid."
        tr = ImageFolder(IMAGENET_TRAIN_DIR, transform=tfm)
        va = ImageFolder(IMAGENET_VAL_DIR, transform=tfm)
        assert tr.class_to_idx == va.class_to_idx
        return tr, va, len(tr.classes)
    if name == "oxfordpet":
        assert os.path.isdir(OXFORD_PET_TRAIN_DIR) and os.path.isdir(OXFORD_PET_VAL_DIR), "Oxford Pet paths invalid."
        tr = ImageFolder(OXFORD_PET_TRAIN_DIR, transform=tfm)
        va = ImageFolder(OXFORD_PET_VAL_DIR, transform=tfm)
        assert tr.class_to_idx == va.class_to_idx
        return tr, va, len(tr.classes)
    raise ValueError(f"Unknown dataset {name}")

@torch.no_grad()
def extract_features(backbone, projector, dataset, batch_size, workers):
    feats_all, labels_all = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device == "cuda"))
    for imgs, labels in tqdm(loader, desc="Features"):
        imgs = imgs.to(device)
        feats = encode_images(backbone, projector, imgs).float().cpu()
        feats_all.append(feats)
        labels_all.append(labels)
    return torch.cat(feats_all).numpy(), torch.cat(labels_all).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", choices=["imagenet", "oxfordpet"], required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--C", type=float, default=0.316)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    backbone, projector = load_backbone_and_projector(args.checkpoint)
    tfm = preprocess()
    train_ds, val_ds, num_classes = load_datasets(args.dataset, tfm)
    print(f"Dataset={args.dataset} Classes={num_classes} Train={len(train_ds)} Val={len(val_ds)}")

    train_feats, train_labels = extract_features(backbone, projector, train_ds, args.batch_size, args.workers)
    val_feats, val_labels = extract_features(backbone, projector, val_ds, args.batch_size, args.workers)

    print(f"len(train_ds) = {len(train_ds)}")
    print(f"train_feats.shape = {train_feats.shape}, train_labels.shape = {train_labels.shape}")
    unique_labels = np.unique(train_labels)
    print(f"#classes in labels = {len(unique_labels)} (expected {num_classes})")

    clf = LogisticRegression(random_state=0, C=args.C, max_iter=args.max_iter,
                             verbose=1, multi_class="auto", solver="lbfgs")
    clf.fit(train_feats, train_labels)
    preds = clf.predict(val_feats)
    acc = (val_labels == preds).mean() * 100
    print(f"Linear probe accuracy (val, top1) = {acc:.3f}")

    # Top-5 accuracy
    probs = clf.predict_proba(val_feats)  # shape (N, num_classes)
    if probs.shape[1] >= 5:
        top5_idx = np.argsort(probs, axis=1)[:, -5:]
        acc5 = np.mean([lbl in row for lbl, row in zip(val_labels, top5_idx)]) * 100
        print(f"Linear probe accuracy (val, top5) = {acc5:.3f}")
    else:
        print("Top-5 not computed (num_classes < 5).")

if __name__ == "__main__":
    main()