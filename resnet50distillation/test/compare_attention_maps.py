import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import CLIPModel, CLIPProcessor
import argparse
import os
from resnet50_non_local import ResNet50_NonLocal,NonLocalBlock



# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

# --- CLIP ViT Attention Extraction (from clip_vit_attention_weight.py) ---
def get_clip_attention_maps(model, processor, image, text, device):
    # Always provide a dummy text input to satisfy CLIP requirements
    inputs = processor(text=["dummy"], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        vision_attentions = outputs.vision_model_output.attentions
    per_layer = []
    for layer_attn in vision_attentions:
        attn_mean = layer_attn.mean(1)
        cls_to_patches = attn_mean[:, 0, 1:]
        per_layer.append(cls_to_patches)
    num_patches = per_layer[-1].shape[-1]
    side = int(num_patches ** 0.5)
    per_layer_maps = []
    for flat in per_layer:
        m = flat.reshape(flat.shape[0], side, side)
        # Normalize before upsampling, matching clip_vit_attention_weight.py
        m = m - m.amin(dim=(1,2), keepdim=True)
        denom = (m.amax(dim=(1,2), keepdim=True) + 1e-6)
        m = m / denom
        per_layer_maps.append(m)
    return per_layer_maps

def upsample_map(m, target_hw):
    import torch.nn.functional as F
    # Match upsampling method and parameters to clip_vit_attention_weight.py
    return F.interpolate(m.unsqueeze(1), size=target_hw, mode="bicubic", align_corners=False).squeeze(1)

def visualize_side_by_side(img, clip_maps, resnet_maps, out_path=None):
    H, W = img.size[1], img.size[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # First row: CLIP ViT
    # The selected CLIP layers are [2, 5, 8, 11] (0-based)
    selected_layers = [2, 5, 8, 11]
    for i in range(4):
        ax = axes[0, i]
        if i < len(clip_maps):
            attn = upsample_map(clip_maps[i].cpu(), (H, W))[0].cpu().numpy()
            # No further normalization here, already normalized before upsampling
            # Use inferno colormap to match clip_vit_attention_weight.py
            ax.imshow(img)
            ax.imshow(attn, cmap='inferno', alpha=0.55)
            layer_num = selected_layers[i] + 1 if i < len(selected_layers) else i + 1
            ax.set_title(f"CLIP Layer {layer_num} (Self-Attn)")
        ax.axis('off')
    # Second row: ResNet50_NonLocal
    for i in range(4):
        ax = axes[1, i]
        if i < len(resnet_maps):
            attn = resnet_maps[i]
            B, N, N2 = attn.shape
            global_attn = attn.mean(dim=1).view(int(N**0.5), int(N**0.5)).cpu().numpy()
            attn_resized = cv2.resize(global_attn, (W, H))
            # Normalize to [0,1] for consistency
            attn_resized = attn_resized - attn_resized.min()
            if attn_resized.max() > 0:
                attn_resized = attn_resized / attn_resized.max()
            ax.imshow(img)
            ax.imshow(attn_resized, cmap='inferno', alpha=0.55)
            ax.set_title(f"NonLocal {i+1}")
        ax.axis('off')
    plt.tight_layout()
    # Always save the figure as 'compare_attention_maps.png' in the script directory
    save_path = out_path if out_path else os.path.join(os.path.dirname(__file__), 'compare_attention_maps.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CLIP ViT and ResNet50 Non-Local Attention Maps")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, default=None, help='Optional path to save the figure')
    args = parser.parse_args()

    img, img_tensor = load_image(args.image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLIP ViT (image only)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_maps = get_clip_attention_maps(clip_model, clip_processor, img, None, device)
    selected_clip_maps = [clip_maps[i] for i in [2, 5, 8, 11] if i < len(clip_maps)]

    # ResNet50_NonLocal
    resnet_model = ResNet50_NonLocal().to(device)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        _, resnet_attn_maps = resnet_model(img_tensor)
    selected_resnet_maps = resnet_attn_maps

    visualize_side_by_side(img, selected_clip_maps, selected_resnet_maps, args.output)