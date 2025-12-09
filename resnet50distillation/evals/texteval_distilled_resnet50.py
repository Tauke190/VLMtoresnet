import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
import clip
import argparse
import os
from pathlib import Path
import sys
import json
import urllib.request

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

IMAGENET_VAL_DIR = os.path.expanduser('~/data/datasets/imagenet/val')
# IMAGENET_VAL_DIR = os.path.expanduser('/home/c3-0/datasets/ImageNet/validation')
OXFORD_PET_VAL_DIR = os.path.expanduser('~/data/datasets/oxford_pet/val')

def find_project_root(start: Path, markers=('utils.py', 'CLIP')):
    p = start.resolve()
    for parent in [p] + list(p.parents):
        utils_file = parent / 'utils.py'
        clip_dir = parent / 'CLIP'
        if utils_file.exists() and clip_dir.exists():
            return parent
    return start.resolve().parent  # fallback

PROJECT_ROOT = find_project_root(Path(__file__).parent)
CLIP_DIR = PROJECT_ROOT / "CLIP"
TEMPLATES_DIR = CLIP_DIR / "dataloaders" / "templates"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(CLIP_DIR))

from utils import (
    zeroshot_classifier,
    get_student_features,
    imagenet_aligned_classnames,
    imagefolder_human_names,
    read_txt,
)

IMAGENET_INDEX_FILENAME = "imagenet_class_index.json"
IMAGENET_INDEX_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json"

def resolve_imagenet_index():
    env_override = os.environ.get("IMAGENET_INDEX_PATH")
    candidates = [
        env_override,
        str(PROJECT_ROOT / IMAGENET_INDEX_FILENAME),
        str(CLIP_DIR / IMAGENET_INDEX_FILENAME),
        str(Path(__file__).parent / IMAGENET_INDEX_FILENAME),
        str(Path.cwd() / IMAGENET_INDEX_FILENAME),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    # Attempt download (silent failure -> fallback)
    target = PROJECT_ROOT / IMAGENET_INDEX_FILENAME
    try:
        print(f"imagenet_class_index.json not found. Attempting download to {target} ...")
        urllib.request.urlretrieve(IMAGENET_INDEX_URL, target)
        if target.is_file():
            print("Downloaded imagenet_class_index.json.")
            return target
    except Exception as e:
        print(f"Download failed: {e}")
    return None  # signal fallback

def evaluate_zero_shot(backbone, projector, loader, zs_weights, device=DEVICE):
    backbone.eval()
    projector.eval()
    zs_weights = zs_weights.to(device=device, dtype=torch.float32)
    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            student_feats = get_student_features(backbone, images)
            proj_feats = projector(student_feats).float()
            proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)
            logits = 100.0 * (proj_feats @ zs_weights)
            _, top5 = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5[:, 0] == labels).sum().item()
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5

def filter_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}

def main():
    parser = argparse.ArgumentParser(description="Final-feature zero-shot evaluation (CLIP->ResNet student).")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'oxfordpet'], default='imagenet')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Resolved PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Templates dir: {TEMPLATES_DIR}")

    if args.dataset == 'imagenet':
        val_dir = IMAGENET_VAL_DIR
        templates_file = TEMPLATES_DIR / "imagenet1k.txt"
    else:
        val_dir = OXFORD_PET_VAL_DIR
        templates_file = TEMPLATES_DIR / "pets.txt"

    print("Loading CLIP ViT-L/14 teacher text encoder only")
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    print(f"Loading dataset from: {val_dir}")
    val_dataset = ImageFolder(root=os.path.expanduser(val_dir), transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Found {len(val_dataset)} images over {len(val_dataset.classes)} classes.")

    templates = read_txt(str(templates_file))
    if args.dataset == 'imagenet':
        index_path = resolve_imagenet_index()
        if index_path:
            print(f"Using ImageNet index at: {index_path}")
            class_names = imagenet_aligned_classnames(val_dataset, str(index_path))
        else:
            print("Warning: imagenet_class_index.json unavailable. Falling back to folder names.")
            class_names = imagefolder_human_names(val_dataset)
    else:
        class_names = imagefolder_human_names(val_dataset)

    print(f"Building zero-shot weights: {len(class_names)} classes, {len(templates)} templates...")
    zs_weights = zeroshot_classifier(class_names, templates, teacher).to(DEVICE)

    print("Instantiating student backbone (ResNet-50) and projector...")
    backbone = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    teacher_dim = teacher.visual.output_dim
    student_dim = backbone.num_features
    projector = nn.Linear(student_dim, teacher_dim).to(DEVICE)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    backbone_sd = filter_state_dict(ckpt.get('backbone_state_dict', {}))
    projector_sd = filter_state_dict(ckpt.get('projector_state_dict', {}))
    backbone.load_state_dict(backbone_sd, strict=True)
    projector.load_state_dict(projector_sd, strict=True)

    print("Running zero-shot evaluation (final feature space)...")
    top1, top5 = evaluate_zero_shot(backbone, projector, val_loader, zs_weights, DEVICE)
    print(f"Zero-shot Accuracy: Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

if __name__ == '__main__':
    main()