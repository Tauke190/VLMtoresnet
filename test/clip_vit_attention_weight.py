#!/usr/bin/env python
"""
Visualize CLIP ViT vision attention maps.

Install deps (CPU ok):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  (or cpu wheel)
    pip install transformers pillow matplotlib tqdm

Usage:
    python clip_vit_attention_weight.py --image path/to/img.jpg --text "a dog" --output-dir outputs
    python clip_vit_attention_weight.py --image img.jpg --text "a cat" --rollout --save-layer-grids

Key features:
  - Per-layer attention (CLS -> patches)
  - Head-averaged maps
  - Attention rollout (per Abnar & Zuidema)
  - Overlay heatmaps on original image
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

def load_image(path, force_rgb=True):
    img = Image.open(path)
    if force_rgb and img.mode != "RGB":
        img = img.convert("RGB")
    return img

def get_attention_maps(vision_attentions, rollout=False):
    """
    vision_attentions: tuple of (num_layers) each (batch, heads, seq, seq)
    Returns:
        per_layer_maps: list[num_layers] of (batch, h, w) tensors in [0,1]
        rollout_maps: (batch, h, w) tensor if rollout=True else None
    """
    num_layers = len(vision_attentions)
    # Use CLS token attention to patch tokens
    per_layer = []
    for layer_attn in vision_attentions:
        # layer_attn: (B, heads, seq, seq)
        b, heads, seq, _ = layer_attn.shape
        # Average heads
        attn_mean = layer_attn.mean(1)  # (B, seq, seq)
        # CLS token index 0 attending to others (exclude CLS itself)
        cls_to_patches = attn_mean[:, 0, 1:]  # (B, num_patches)
        per_layer.append(cls_to_patches)  # still flat
    # Determine spatial size from last
    num_patches = per_layer[-1].shape[-1]
    side = int(num_patches ** 0.5)
    assert side * side == num_patches, "Patch count not square; unexpected model configuration."
    per_layer_maps = []
    for flat in per_layer:
        m = flat.reshape(flat.shape[0], side, side)
        # Normalize each map to [0,1]
        m = m - m.amin(dim=(1,2), keepdim=True)
        denom = (m.amax(dim=(1,2), keepdim=True) + 1e-6)
        m = m / denom
        per_layer_maps.append(m)
    rollout_map = None
    if rollout:
        # Attention rollout: multiply (I + A_headAvg) normalized across layers
        # Based on: Attention Rollout (Abnar & Zuidema 2020)
        with torch.no_grad():
            # Prepare list of head-averaged attention (B, seq, seq)
            head_avgs = [a.mean(1) for a in vision_attentions]
            # Remove residual? Use identity addition then row-normalize
            rollout_mat = None
            for A in head_avgs:
                A = A + torch.eye(A.size(-1), device=A.device).unsqueeze(0)
                A = A / A.sum(-1, keepdim=True)
                rollout_mat = A if rollout_mat is None else torch.bmm(A, rollout_mat)
            # CLS to patches after rollout
            cls_rollout = rollout_mat[:, 0, 1:]  # (B, num_patches)
            r = cls_rollout.reshape(cls_rollout.shape[0], side, side)
            r = r - r.amin(dim=(1,2), keepdim=True)
            r = r / (r.amax(dim=(1,2), keepdim=True) + 1e-6)
            rollout_map = r
    return per_layer_maps, rollout_map

def upsample_maps(maps, target_hw):
    """
    maps: list of (B,H,W) tensors or single tensor (B,H,W)
    target_hw: (H, W)
    Returns list/tensor (B, targetH, targetW)
    """
    if isinstance(maps, list):
        return [F.interpolate(m.unsqueeze(1), size=target_hw, mode="bicubic", align_corners=False).squeeze(1) for m in maps]
    else:
        return F.interpolate(maps.unsqueeze(1), size=target_hw, mode="bicubic", align_corners=False).squeeze(1)

def overlay_and_save(image_pil, heatmap, out_path, cmap='inferno', alpha=0.5):
    """
    heatmap: (H,W) numpy in [0,1]
    """
    plt.figure(figsize=(4,4), dpi=150)
    plt.imshow(image_pil)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_grid(per_layer_maps, image_pil, out_path, cols=6, cmap='inferno', alpha=0.5):
    b = per_layer_maps[0].shape[0]
    assert b == 1, "Grid saver assumes batch=1."
    layers = len(per_layer_maps)
    rows = (layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    base = np.array(image_pil)
    for i in range(rows*cols):
        ax = axes[i]
        ax.axis('off')
        if i < layers:
            hm = per_layer_maps[i][0].cpu().numpy()
            ax.imshow(base)
            ax.imshow(hm, cmap=cmap, alpha=alpha)
            ax.set_title(f"L{i}")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--text", required=True, help="Prompt text")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="clip_attention_out")
    ap.add_argument("--rollout", action="store_true", help="Compute attention rollout map")
    ap.add_argument("--save-layer-grids", action="store_true", help="Save a grid of all layer maps")
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--layers", type=str, default="all", help="Comma list of layer indices or 'all'")
    ap.add_argument("--no-individual", action="store_true", help="Skip saving per-layer individual overlays")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading model {args.model} on {device} ...")
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.to(device)
    model.eval()

    image = load_image(args.image)
    inputs = processor(text=[args.text],
                       images=[image],
                       return_tensors="pt",
                       padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        # vision attentions is tuple (num_layers) each (B, heads, seq, seq)
        vision_attentions = outputs.vision_model_output.attentions

    per_layer_maps, rollout_map = get_attention_maps(vision_attentions, rollout=args.rollout)

    # Filter layers if specified
    if args.layers != "all":
        wanted = [int(x) for x in args.layers.split(",") if x.strip() != ""]
        per_layer_maps = [m for i, m in enumerate(per_layer_maps) if i in wanted]

    # Upsample to image size
    target_hw = image.size[1], image.size[0]  # (H,W)
    upsampled = upsample_maps(per_layer_maps, target_hw)
    if rollout_map is not None:
        rollout_up = upsample_maps(rollout_map, target_hw)


    # Save attention maps after every 3 layers (layers 3, 6, 9, 12)
    selected_layers = [2, 5, 8, 11]  # 0-based indices
    for i in selected_layers:
        if i < len(upsampled):
            hm = upsampled[i][0].cpu().numpy()
            out_path = Path(args.output_dir) / f"attention_layer_{i+1}.png"
            overlay_and_save(image, hm, out_path, cmap=args.cmap, alpha=args.alpha)

    # Save individual layers as before
    if not args.no_individual:
        for i, m in enumerate(upsampled):
            hm = m[0].cpu().numpy()
            out_path = Path(args.output_dir) / f"attention_layer_{i}.png"
            overlay_and_save(image, hm, out_path, cmap=args.cmap, alpha=args.alpha)

    # Save grid
    if args.save_layer_grids:
        save_grid(upsampled, image, Path(args.output_dir) / "layers_grid.png", cmap=args.cmap, alpha=args.alpha)

    # Save rollout
    if rollout_map is not None:
        hm_r = rollout_up[0].cpu().numpy()
        overlay_and_save(image, hm_r, Path(args.output_dir) / "attention_rollout.png", cmap=args.cmap, alpha=args.alpha)

    # Also save raw numpy arrays
    npy_dir = Path(args.output_dir) / "npy"
    npy_dir.mkdir(exist_ok=True)
    for i, m in enumerate(per_layer_maps):
        np.save(npy_dir / f"layer_{i}.npy", m[0].cpu().numpy())
    if rollout_map is not None:
        np.save(npy_dir / "rollout.npy", rollout_map[0].cpu().numpy())

    print("Done. Outputs in", args.output_dir)

if __name__ == "__main__":
    main()