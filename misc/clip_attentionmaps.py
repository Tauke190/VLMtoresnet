"""
Visualize attention maps from the last block (layer4) of CLIP ViT-L/14
using the grouped model.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CLIP.clip.model_grouped import build_model, get_layer4_attention_maps
from CLIP import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load grouped CLIP model from pretrained weights ---
print("Loading CLIP ViT-L/14 (grouped)...")
model_path = os.path.expanduser("~/.cache/clip/ViT-L-14.pt")
if not os.path.exists(model_path):
    # Download via clip if not cached
    clip.load("ViT-L/14", device="cpu", jit=False)

state_dict = torch.jit.load(model_path, map_location="cpu").state_dict()
model = build_model(state_dict).to(device)
model.eval()

_, preprocess = clip.load("ViT-L/14", device="cpu", jit=False)

# --- Prepare input ---
# Use a real image if provided, otherwise use a random tensor
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img = Image.open(sys.argv[1]).convert("RGB")
    images = preprocess(img).unsqueeze(0).to(device)
    print(f"Using image: {sys.argv[1]}")
else:
    images = torch.randn(1, 3, 224, 224).to(device)
    print("Using random input (pass an image path as argument for real images)")

# --- Extract attention maps from layer4 ---
print("Extracting attention maps from layer4 (last block)...")
attn_maps, output = get_layer4_attention_maps(model, images)

print(f"Number of layer4 blocks: {len(attn_maps)}")
for i, attn in enumerate(attn_maps):
    print(f"  Block {i} attention shape: {attn.shape}")  # [B, seq_len, seq_len]

# --- Visualize ---
save_dir = "attention_maps_clip_layer4"
os.makedirs(save_dir, exist_ok=True)

for block_idx, attn in enumerate(attn_maps):
    # attn shape: [B, seq_len, seq_len] (averaged over heads by nn.MultiheadAttention)
    attn_np = attn[0].cpu().float().numpy()  # first image in batch

    # seq_len = num_patches + 1 (class token)
    seq_len = attn_np.shape[0]
    grid_size = int(np.sqrt(seq_len - 1))  # e.g. 16 for ViT-L/14

    # --- Plot 1: CLS token attention over spatial patches ---
    cls_attn = attn_np[0, 1:]  # CLS attending to all patches
    cls_attn_grid = cls_attn.reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cls_attn_grid, cmap="viridis")
    axes[0].set_title(f"Layer4 Block {block_idx}: CLS → patches")
    axes[0].axis("off")

    # --- Plot 2: Mean attention (all tokens) ---
    mean_attn = attn_np[:, 1:].mean(axis=0)  # average over query tokens
    mean_attn_grid = mean_attn.reshape(grid_size, grid_size)

    axes[1].imshow(mean_attn_grid, cmap="viridis")
    axes[1].set_title(f"Layer4 Block {block_idx}: Mean attention")
    axes[1].axis("off")

    plt.tight_layout()
    fname = os.path.join(save_dir, f"clip_layer4_block{block_idx}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

# --- Plot full attention matrix ---
for block_idx, attn in enumerate(attn_maps):
    attn_np = attn[0].cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(attn_np, cmap="viridis")
    ax.set_title(f"CLIP Layer4 Block {block_idx} — Full attention matrix")
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    fname = os.path.join(save_dir, f"clip_layer4_block{block_idx}_full.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

print(f"\nAll maps saved to {save_dir}/")
print(f"Image features shape: {output.shape}")
