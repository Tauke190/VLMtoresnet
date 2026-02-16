import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from CLIP.dataloaders import DiffisionImages


def load_captions(root):
    captions_2 = os.path.join(root, "caption_2k.txt")
    captions_5 = os.path.join(root, "caption_5k.txt")

    captions = []
    for path in (captions_2, captions_5):
        with open(path, "r") as f:
            captions.extend(line.strip() for line in f if line.strip())
    return captions


def evaluate_diffusion_dataset(loader, model, text_features, device):
    text_features = text_features.to(device)
    text_features = F.normalize(text_features.float(), dim=-1)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            
            labels = labels.to(device) 

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features.float(), dim=-1)

            logits = 100.0 * image_features @ text_features.T  # [B, num_classes]

            maxk = min(5, logits.size(1))
            _, pred = logits.topk(maxk, 1, True, True)  # [B, maxk]
            pred = pred.t() 

            correct = pred.eq(labels.view(1, -1).expand_as(pred))  # [maxk, B]

            correct_top1 += correct[:1].reshape(-1).float().sum().item()
            correct_top5 += correct[:maxk].reshape(-1).float().sum().item()
            total += labels.size(0)

    top1 = 100.0 * correct_top1 / max(total, 1)
    top5 = 100.0 * correct_top5 / max(total, 1)
    return top1, top5


def main(train=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    root = "/mnt/SSD2/Diffision_images"
    all_captions = load_captions(root)
    total_captions = len(all_captions)

    if train:
        captions_slice = all_captions[:6000]
    else:
        captions_slice = all_captions[-1000:]

    print(f"{'Train' if train else 'Test'} captions: {len(captions_slice)}")
    text_tokens = clip.tokenize(captions_slice).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)

    dataset = DiffisionImages(
        root=root,
        transform=preprocess,
        train=train
    )
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    top1, top5 = evaluate_diffusion_dataset(loader, clip_model, text_features, device)

    print(f"CLIP Zero-Shot {'Train' if train else 'Test'} Results =====")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")


main(train=True)