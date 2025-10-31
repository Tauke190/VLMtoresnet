import torch
import timm
import clip
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16

parser = argparse.ArgumentParser(description="Evaluate cosine similarity between student and CLIP logits")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
parser.add_argument("--dataset", type=str, required=True, choices=["imagenet", "oxfordpet"], help="Dataset type")
args = parser.parse_args()

if args.dataset == "imagenet":
    VAL_DIR = '/home/av354855/data/datasets/imagenet/val'
    # VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'
elif args.dataset == "oxfordpet":
    VAL_DIR = '/home/av354855/data/datasets/oxford_pet/val'
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

# Load student backbone and projector
backbone = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
backbone.load_state_dict(checkpoint['backbone_state_dict'])
backbone.eval()

teacher, preprocess = clip.load("ViT-L/14", device=DEVICE)
teacher.eval()

student_feature_dim = backbone.num_features
teacher_feature_dim = teacher.visual.output_dim
projector = nn.Linear(student_feature_dim, teacher_feature_dim).to(DEVICE)
projector.load_state_dict(checkpoint['projector_state_dict'])
projector.eval()

# Load validation dataset
val_dataset = ImageFolder(root=VAL_DIR, transform=preprocess)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def get_student_features(backbone, images):
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    return pooled_features

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(DEVICE)
        # CLIP logits
        clip_features = teacher.encode_image(images).float()
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        # Student logits
        student_features = get_student_features(backbone, images)
        projected_features = projector(student_features)
        projected_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(projected_features, clip_features, dim=-1)
        print("Cosine similarity between student and CLIP logits:", similarity.cpu().numpy())
        break  # Remove break to process all batches