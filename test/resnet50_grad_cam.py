#!/usr/bin/env python
"""
Grad-CAM for ResNet50.

Usage:
  python resnet50_grad_cam.py --image dog.jpg
  python resnet50_grad_cam.py --image dog.jpg --class-index 243   # (optional specific class)

Dependencies:
  pip install torch torchvision pillow matplotlib
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def get_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register()

    def _register(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_backward_hook(bwd_hook)

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Grad-CAM weighting
        grads = self.gradients  # (B,C,H,W)
        acts = self.activations # (B,C,H,W)
        weights = grads.mean(dim=(2,3), keepdim=True)  # (B,C,1,1)
        cam = (weights * acts).sum(dim=1)  # (B,H,W)
        cam = F.relu(cam)
        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        cam = F.interpolate(cam.unsqueeze(1), size=(224,224), mode="bilinear", align_corners=False).squeeze(1)
        return cam, class_idx, logits.softmax(dim=1)

def cam_to_colormap(cam_tensor, cmap_name='jet'):
    """
    cam_tensor: (H,W) in [0,1]
    Returns uint8 RGB heatmap (H,W,3)
    """
    cam_np = cam_tensor.detach().cpu().numpy()
    colored = cm.get_cmap(cmap_name)(cam_np)[:, :, :3]  # drop alpha
    colored = (colored * 255).astype('uint8')
    return colored

def blend_heatmap_on_image(img_pil, heatmap_rgb, alpha=0.45):
    """
    heatmap_rgb: uint8 (H,W,3)
    Returns blended PIL image.
    """
    img = np.array(img_pil.resize((heatmap_rgb.shape[1], heatmap_rgb.shape[0])))
    blended = (alpha * heatmap_rgb + (1 - alpha) * img).astype('uint8')
    return Image.fromarray(blended)

def save_overlay_set(base_img_pil, cam_map, output_prefix, alpha=0.45, cmap='jet'):
    """
    cam_map: (H,W) torch tensor [0,1]
    Saves:
      {prefix}_original.png
      {prefix}_heatmap.png        (colored heatmap alone)
      {prefix}_overlay.png        (blended original + heatmap)
      {prefix}_matplot.png        (Matplotlib version)
    """
    hmap_rgb = cam_to_colormap(cam_map, cmap_name=cmap)
    blended = blend_heatmap_on_image(base_img_pil, hmap_rgb, alpha=alpha)

    base_img_pil.save(f"{output_prefix}_original.png")
    Image.fromarray(hmap_rgb).save(f"{output_prefix}_heatmap.png")
    blended.save(f"{output_prefix}_overlay.png")

    # Optional Matplotlib overlay (as before)
    fig = overlay(base_img_pil, cam_map, alpha=alpha, cmap=cmap)
    fig.savefig(f"{output_prefix}_matplot.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Image path")
    ap.add_argument("--class-index", type=int, default=None, help="Optional class index to visualize")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--cmap", default="jet", help="Colormap")
    ap.add_argument("--output", default="dog_gradcam_overlay.png", help="Output overlay filename")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    img = load_image(args.image)                      # original aspect ratio
    orig_w, orig_h = img.size

    preprocess = get_preprocess()                     # produces 224x224 tensor
    tensor = preprocess(img).unsqueeze(0).to(device)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    target_layer = model.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)
    cam, cls_idx, probs = gradcam(tensor, class_idx=args.class_index)
    print(f"Class index: {cls_idx}, prob: {probs[0, cls_idx].item():.4f}")

    # Resize CAM (224x224) to original image size (preserve aspect ratio of original by stretching if not square)
    cam_full = F.interpolate(cam.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)[0]

    # Create colored heatmap
    heatmap_rgb = cam_to_colormap(cam_full, cmap_name=args.cmap)
    # Blend with original (original aspect ratio)
    blended = blend_heatmap_on_image(img, heatmap_rgb, alpha=args.alpha)
    blended.save(args.output)
    print(f"Saved overlay: {args.output}")

if __name__ == "__main__":
    main()