import os
import math
import json
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

def get_backbone(model: nn.Module) -> nn.Module:
    # Works for most torchvision resnets (drops final FC)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return nn.Sequential(*list(model.children())[:-1])  # keeps avgpool
    # Fallback: assume model already outputs features
    return model

@torch.no_grad()
def save_split_features(
    model: nn.Module,
    dataloader,
    out_dir: str,
    split_name: str = "train",
    device: str = None,
    shard_size: int = 4096,
    use_amp: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    backbone = get_backbone(model).to(device).eval()

    shard_feats, shard_labels = [], []
    total = 0
    shard_idx = 0
    feat_dim = None

    for images, labels, *rest in dataloader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            feats = backbone(images)
        # Flatten to (N, D)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        feats = feats.float().cpu().numpy()
        labels = labels.cpu().numpy()

        if feat_dim is None:
            feat_dim = feats.shape[1]

        shard_feats.append(feats)
        shard_labels.append(labels)
        total += feats.shape[0]

        # Flush to disk in shards
        cur_count = sum(x.shape[0] for x in shard_feats)
        if cur_count >= shard_size:
            X = np.concatenate(shard_feats, axis=0)
            y = np.concatenate(shard_labels, axis=0)
            shard_path = os.path.join(out_dir, f"{split_name}_{shard_idx:04d}.npz")
            np.savez_compressed(shard_path, X=X, y=y)
            shard_idx += 1
            shard_feats, shard_labels = [], []
            # Optional: release GPU memory promptly
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Remainder
    if shard_feats:
        X = np.concatenate(shard_feats, axis=0)
        y = np.concatenate(shard_labels, axis=0)
        shard_path = os.path.join(out_dir, f"{split_name}_{shard_idx:04d}.npz")
        np.savez_compressed(shard_path, X=X, y=y)

    # Save simple metadata
    meta = {
        "split": split_name,
        "feature_dim": int(feat_dim or -1),
        "count_estimate": int(total),
        "shard_size": int(shard_size),
    }
    with open(os.path.join(out_dir, f"{split_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{split_name}] saved ~{total} samples to {out_dir}")