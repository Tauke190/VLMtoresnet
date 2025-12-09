import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

# -------------------------------
# Non-Local Block
# -------------------------------
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_map = None  # store attention map

    def forward(self, x):
        B, C, H, W = x.shape
        theta_x = self.theta(x).view(B, C//2, -1).permute(0, 2, 1)  # B, N, C'
        phi_x = self.phi(x).view(B, C//2, -1)                        # B, C', N
        f = torch.bmm(theta_x, phi_x)                                 # B, N, N
        f_div_C = self.softmax(f)
        self.attention_map = f_div_C.detach()                         # store for visualization

        g_x = self.g(x).view(B, C//2, -1).permute(0, 2, 1)           # B, N, C'
        y = torch.bmm(f_div_C, g_x)                                  # B, N, C'
        y = y.permute(0, 2, 1).contiguous().view(B, C//2, H, W)
        y = self.out_conv(y)
        return x + y

# -------------------------------
# Modified ResNet-50 with Non-local blocks
# -------------------------------
class ResNet50_NonLocal(nn.Module):
    def __init__(self):
        super(ResNet50_NonLocal, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.nl1 = NonLocalBlock(256)   # after layer1
        self.nl2 = NonLocalBlock(512)   # after layer2
        self.nl3 = NonLocalBlock(1024)  # after layer3
        self.nl4 = NonLocalBlock(2048)  # after layer4

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.nl1(x)
        x1 = x  # store feature for attention map

        x = self.resnet.layer2(x)
        x = self.nl2(x)
        x2 = x

        x = self.resnet.layer3(x)
        x = self.nl3(x)
        x3 = x

        x = self.resnet.layer4(x)
        x = self.nl4(x)
        x4 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x, [self.nl1.attention_map, self.nl2.attention_map, self.nl3.attention_map, self.nl4.attention_map]

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # B, C, H, W
    return img, img_tensor

# -------------------------------
# Attention visualization
# -------------------------------
def visualize_attention(img, attn_map, block_name="block"):
    B, N, N = attn_map.shape
    H, W = img.size[1], img.size[0]

    # global attention: average over all query positions
    global_attn = attn_map.mean(dim=1).view(int(N**0.5), int(N**0.5)).cpu().numpy()
    attn_resized = cv2.resize(global_attn, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    img_np = np.array(img)
    overlay = 0.6 * img_np + 0.4 * heatmap
    plt.figure(figsize=(6,6))
    plt.imshow(np.uint8(overlay))
    plt.title(f"Attention Map - {block_name}")
    plt.axis('off')
    plt.show()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize Non-Local Attention Maps for ResNet50")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    img, img_tensor = load_image(args.image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_NonLocal().to(device)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output, attn_maps = model(img_tensor)

    # visualize original image and attention maps for all non-local blocks in the same figure
    plt.figure(figsize=(20, 4))
    # Show original image first
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    for idx, attn in enumerate(attn_maps):
        B, N, N = attn.shape
        H, W = img.size[1], img.size[0]
        global_attn = attn.mean(dim=1).view(int(N**0.5), int(N**0.5)).cpu().numpy()
        attn_resized = cv2.resize(global_attn, (W, H))
        # Make heatmap more prominent by increasing its weight
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.4, heatmap, 0.6, 0)  # more heatmap, less image

        plt.subplot(1, 5, idx + 2)
        plt.imshow(np.uint8(overlay))
        plt.title(f"NonLocal_{idx+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
