import os
import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    timm = None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hardcoded dataset directories
IMAGENET_TRAIN_DIR = os.path.expanduser('~/data/datasets/imagenet/train')
IMAGENET_VAL_DIR   = os.path.expanduser('~/data/datasets/imagenet/val')

OXFORD_PET_TRAIN_DIR = os.path.expanduser('~/data/datasets/oxford_pet/train')
OXFORD_PET_VAL_DIR   = os.path.expanduser('~/data/datasets/oxford_pet/val')

# Add more datasets similarly if needed (e.g., flowers)
# FLOWERS_TRAIN_DIR = ...
# FLOWERS_VAL_DIR   = ...

def preprocess():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

class StudentEncoder(torch.nn.Module):
    def __init__(self, backbone, projector, normalize=True):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.normalize = normalize

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features") and hasattr(self.backbone, "global_pool"):
            fm = self.backbone.forward_features(images)
            pooled = self.backbone.global_pool(fm).flatten(1)
        else:
            out = self.backbone(images)
            pooled = out if out.ndim == 2 else torch.flatten(out, 1)
        feats = self.projector(pooled)
        if self.normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

def build_projector(sd: dict) -> torch.nn.Module:
    # Simple: single Linear or sequential numbered layers (0.weight, 1.weight,...)
    if "weight" in sd and sd["weight"].ndim == 2:
        out_d, in_d = sd["weight"].shape
        layer = torch.nn.Linear(in_d, out_d, bias=("bias" in sd))
        layer.load_state_dict(sd, strict=False)
        return layer
    layer_keys = sorted([k for k in sd if k.endswith(".weight") and k.split(".")[0].isdigit()],
                        key=lambda k: int(k.split(".")[0]))
    modules = []
    for k in layer_keys:
        idx = k.split(".")[0]
        w = sd[k]
        out_d, in_d = w.shape
        bias_key = f"{idx}.bias"
        lin = torch.nn.Linear(in_d, out_d, bias=(bias_key in sd))
        modules.append(lin)
        if k != layer_keys[-1]:
            modules.append(torch.nn.ReLU(inplace=True))
    proj = torch.nn.Sequential(*modules) if modules else torch.nn.Identity()
    proj.load_state_dict(sd, strict=False)
    return proj

def load_student(checkpoint_path: str) -> StudentEncoder:
    assert timm is not None, "Install timm: pip install timm"
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    bb_sd = ckpt.get("backbone_state_dict") or ckpt.get("backbone")
    pj_sd = ckpt.get("projector_state_dict") or ckpt.get("projector")
    assert bb_sd is not None and pj_sd is not None, "Checkpoint must have backbone_state_dict and projector_state_dict."
    bb_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in bb_sd.items() }
    backbone = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="avg")
    load_info = backbone.load_state_dict(bb_sd, strict=False)
    print(f"Backbone loaded. Missing={len(load_info.missing_keys)} Unexpected={len(load_info.unexpected_keys)}")
    projector = build_projector(pj_sd)
    model = StudentEncoder(backbone, projector).to(device).eval()
    return model

def load_datasets(dataset: str, tfm):
    if dataset == "imagenet":
        assert os.path.isdir(IMAGENET_TRAIN_DIR) and os.path.isdir(IMAGENET_VAL_DIR), "ImageNet paths invalid."
        train = ImageFolder(IMAGENET_TRAIN_DIR, transform=tfm)
        val = ImageFolder(IMAGENET_VAL_DIR, transform=tfm)
        assert train.class_to_idx == val.class_to_idx
        return train, val, len(train.classes)
    if dataset == "oxford_pet":
        # Using ImageFolder style (train/val folder with class subdirs)
        assert os.path.isdir(OXFORD_PET_TRAIN_DIR) and os.path.isdir(OXFORD_PET_VAL_DIR), "Oxford Pet paths invalid."
        train = ImageFolder(OXFORD_PET_TRAIN_DIR, transform=tfm)
        val = ImageFolder(OXFORD_PET_VAL_DIR, transform=tfm)
        assert train.class_to_idx == val.class_to_idx
        return train, val, len(train.classes)
    raise ValueError(f"Unknown dataset {dataset}")

@torch.no_grad()
def extract_features(model, dataset, batch_size, workers):
    feats_list, labels_list = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device == "cuda"))
    for imgs, lbls in tqdm(loader, desc="Features"):
        imgs = imgs.to(device)
        f = model.encode_image(imgs).float().cpu()
        feats_list.append(f)
        labels_list.append(lbls)
    return torch.cat(feats_list).numpy(), torch.cat(labels_list).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--dataset", choices=["imagenet", "oxford_pet"], required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--C", type=float, default=0.316)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    model = load_student(args.checkpoint)
    tfm = preprocess()

    train_ds, val_ds, num_classes = load_datasets(args.dataset, tfm)
    print(f"Dataset={args.dataset} Classes={num_classes} Train={len(train_ds)} Val={len(val_ds)}")

    train_feats, train_labels = extract_features(model, train_ds, args.batch_size, args.workers)
    val_feats, val_labels = extract_features(model, val_ds, args.batch_size, args.workers)

    clf = LogisticRegression(random_state=0, C=args.C, max_iter=args.max_iter,
                             verbose=1, multi_class="auto", solver="lbfgs")
    clf.fit(train_feats, train_labels)
    preds = clf.predict(val_feats)
    acc = (val_labels == preds).mean() * 100
    print(f"Linear probe accuracy (val) = {acc:.3f}")

if __name__ == "__main__":
    main()