"""
Visualize attention maps from the last block (network.7) of FastViT SA36.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fastvit_proposed import fastvit_sa36_lrtokens
from models.modules.attention_extractor import AttentionMapExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load FastViT model ---
print("Loading FastViT SA36 lrtokens...")
model = fastvit_sa36_lrtokens()
model = model.to(device)
model.eval()

# Load checkpoint if provided
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    ckpt = torch.load(sys.argv[2], map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {sys.argv[2]}")

# --- List attention layers ---
extractor = AttentionMapExtractor(model)
print(f"Attention layers found: {extractor.list_attention_layers()}")

# --- Prepare input ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img = Image.open(sys.argv[1]).convert("RGB")
    images = preprocess(img).unsqueeze(0).to(device)
    print(f"Using image: {sys.argv[1]}")
else:
    images = torch.randn(1, 3, 224, 224).to(device)
    print("Using random input (pass an image path as argument for real images)")

# --- Extract attention maps ---
print("Extracting attention maps from last stage...")
output, attn_maps = extractor(images)

# Filter to last stage only (network.7)
last_stage_maps = {k: v for k, v in attn_maps.items() if "network.7" in k}
print(f"Last stage attention layers: {list(last_stage_maps.keys())}")
for name, attn in last_stage_maps.items():
    print(f"  {name}: shape {attn.shape}")  # [B, num_heads, N, N]

# --- Visualize ---
save_dir = "attention_maps_fastvit_last"
os.makedirs(save_dir, exist_ok=True)

for layer_name, attn in last_stage_maps.items():
    # attn shape: [B, num_heads, N, N] where N = H*W (spatial tokens)
    attn_np = attn[0].cpu().float().numpy()  # [num_heads, N, N]
    num_heads, N, _ = attn_np.shape
    grid_size = int(np.sqrt(N))  # e.g. 7 for 7x7 spatial

    short_name = layer_name.replace("network.", "net").replace(".token_mixer", "")

    # --- Plot 1: Per-head mean attention ---
    fig, axes = plt.subplots(2, min(num_heads, 8), figsize=(24, 6))
    if num_heads == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for h in range(min(num_heads, 8)):
        head_attn = attn_np[h]  # [N, N]
        # Mean attention received by each spatial location
        mean_received = head_attn.mean(axis=0).reshape(grid_size, grid_size)
        axes[0, h].imshow(mean_received, cmap="viridis")
        axes[0, h].set_title(f"Head {h} (received)")
        axes[0, h].axis("off")

        # Mean attention given by each spatial location
        mean_given = head_attn.mean(axis=1).reshape(grid_size, grid_size)
        axes[1, h].imshow(mean_given, cmap="viridis")
        axes[1, h].set_title(f"Head {h} (given)")
        axes[1, h].axis("off")

    plt.suptitle(f"FastViT {layer_name} — Per-head attention")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"{short_name}_heads.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

    # --- Plot 2: Head-averaged attention map ---
    avg_attn = attn_np.mean(axis=0)  # [N, N]
    avg_received = avg_attn.mean(axis=0).reshape(grid_size, grid_size)
    avg_given = avg_attn.mean(axis=1).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(avg_received, cmap="viridis")
    axes[0].set_title("Avg attention received")
    axes[0].axis("off")

    axes[1].imshow(avg_given, cmap="viridis")
    axes[1].set_title("Avg attention given")
    axes[1].axis("off")

    axes[2].imshow(avg_attn, cmap="viridis")
    axes[2].set_title("Full attention matrix (head avg)")
    axes[2].set_xlabel("Key")
    axes[2].set_ylabel("Query")

    plt.suptitle(f"FastViT {layer_name} — Head-averaged")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"{short_name}_avg.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

# --- Clean up ---
extractor.remove_hooks()

print(f"\nAll maps saved to {save_dir}/")
print(f"Model output shape: {output.shape}")
