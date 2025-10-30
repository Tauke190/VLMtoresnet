import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import clip
import argparse
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# Hardcoded paths and parameters
VAL_DIR = '~/data/datasets/imagenet/val'
# For coding server
# VAL_DIR = os.path.expanduser('~/data/datasets/imagenet/val')
PROMPT_FILE = '../prompt/imagenet1k.txt'
NUM_TEMPLATES = 2

def get_student_features(backbone, images):
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    return pooled_features

def load_prompts_from_file(filepath):
    with open(filepath, 'r') as f:
        templates = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(templates)} templates from {filepath}.")
    return templates

def zeroshot_validate_student(backbone, projector, class_names, val_loader, teacher, templates, device=DEVICE):
    prompts = [template.format(name) for name in class_names for template in templates]
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = teacher.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    num_templates = len(templates)
    num_classes = len(class_names)
    text_features = text_features.view(num_classes, num_templates, -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    top1_correct = 0
    top5_correct = 0
    total = 0
    backbone.eval()
    projector.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = get_student_features(backbone, images)
            student_features = projector(features)
            student_features = student_features / student_features.norm(dim=-1, keepdim=True)
            logits = student_features @ text_features.t()
            _, top5_preds = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy

def filter_state_dict(state_dict):
    # Remove keys containing 'total_ops' or 'total_params'
    return {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}

def main():
    parser = argparse.ArgumentParser(description="Zero-shot validation for distilled ResNet-50 student")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # Load teacher model and preprocessing
    teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Load validation dataset
    print(f"Loading validation dataset from: {VAL_DIR}")
    val_dataset = ImageFolder(root=VAL_DIR, transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    class_names = val_dataset.classes
    print(f"Found {len(class_names)} classes in validation set.")

    # Load prompt templates
    templates = load_prompts_from_file(PROMPT_FILE)
    templates = templates[:NUM_TEMPLATES]

    # Load student backbone and projector
    print("Loading student backbone (ResNet-50)...")
    backbone = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    teacher_feature_dim = teacher.visual.output_dim
    student_feature_dim = backbone.num_features
    projector = nn.Linear(teacher_feature_dim, student_feature_dim).to(DEVICE)

    # Load checkpoint and filter state_dict
    print(f"Loading checkpoint from {args.checkpoint} ...")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    backbone_state_dict = filter_state_dict(checkpoint['backbone_state_dict'])
    projector_state_dict = filter_state_dict(checkpoint['projector_state_dict'])
    backbone.load_state_dict(backbone_state_dict)
    projector.load_state_dict(projector_state_dict)

    # Run zero-shot validation
    print("Running zero-shot validation...")
    top1, top5 = zeroshot_validate_student(backbone, projector, class_names, val_loader, teacher, templates, DEVICE)
    print(f"Zero-shot Validation Accuracy: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")

if __name__ == '__main__':
    main()