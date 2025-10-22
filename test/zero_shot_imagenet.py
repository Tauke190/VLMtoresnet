import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from mmengine.config import Config
from mmcls.models import build_classifier
from mmengine.runner import load_checkpoint
import clip
import json
from sklearn.metrics import accuracy_score

# ---------------- CONFIGURATION ----------------
# --- Your Custom ResNet Model ---
config_path = 'configs/imagenet/resnet50_8xb32_in1k_strong_aug_coslr_300.py'
checkpoint_path = 'resnet50_scalekd_e300_new.pth'

# --- Dataset Paths ---
# ✅ CHANGE 1: Point this to the validation folder of the ImageNet dataset
imagenet_val_path = '~/data/datasets/ImageNet/val'
imagenet_class_index_path = 'imagenet_class_index.json'

# --- Script Settings ---
BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- STEP 1: LOAD MODEL AND EXTRACT KNOWLEDGE ----------------

print("Loading custom ResNet-50 Image Classifier...")
cfg = Config.fromfile(config_path)
model = build_classifier(cfg.model)
load_checkpoint(model, checkpoint_path, map_location=device)
model.to(device)
model.eval()

print("Extracting learned ImageNet embeddings from the model's head...")
with torch.no_grad():
    imagenet_knowledge_vectors = model.head.fc.weight.clone().float() # Shape: [1000, 2048]

# ---------------- STEP 2: BUILD THE TEXT-BASED KNOWLEDGE BRIDGE ----------------

print("Loading CLIP and building text-based knowledge bridge...")
clip_model, _ = clip.load("ViT-B/32", device=device)

# Define the source and target class names
with open(imagenet_class_index_path, 'r') as f:
    class_idx = json.load(f)
# Source classes are ImageNet
source_classes = [class_idx[str(k)][1].replace("_", " ") for k in sorted([int(i) for i in class_idx.keys()])]
# ✅ CHANGE 2: Target classes are also ImageNet for this test
target_classes = source_classes

# Create text features for both sets of classes
with torch.no_grad():
    # Source (ImageNet) text features
    source_text_tokens = clip.tokenize([f"a photo of a {c}" for c in source_classes]).to(device)
    source_text_features = clip_model.encode_text(source_text_tokens)
    source_text_features = F.normalize(source_text_features, p=2, dim=1)
    
    # Target (also ImageNet) text features
    # In this specific case, this is redundant, but we keep the logic for clarity
    target_text_tokens = clip.tokenize([f"a photo of a {c}" for c in target_classes]).to(device)
    target_text_features = clip_model.encode_text(target_text_tokens)
    target_text_features = F.normalize(target_text_features, p=2, dim=1)

# Calculate the similarity matrix (the bridge) between target and source classes
# Shape: [1000, 1000]. Should ideally be close to an Identity matrix.
text_similarity_bridge = target_text_features @ source_text_features.T

# ---------------- STEP 3: SYNTHESIZE IMAGENET CLASS EMBEDDINGS ----------------

print("Synthesizing new ImageNet class embeddings...")
# Use the bridge to "reconstruct" the ImageNet knowledge vectors
# [1000, 1000] @ [1000, 2048] -> [1000, 2048]

# ✅ CHANGE 3 (CRITICAL FIX): Cast the bridge to .float() to match the knowledge vectors' type
synthesized_embeddings = text_similarity_bridge.float() @ imagenet_knowledge_vectors

# Normalize the newly created embeddings for accurate comparison
synthesized_embeddings = F.normalize(synthesized_embeddings, p=2, dim=1)

# ---------------- STEP 4: EVALUATE ON THE IMAGENET DATASET ----------------
print("\nEvaluating on the ImageNet validation set...")
from torchvision import transforms

# Use the same transforms your model was trained with
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_image_features(img_tensors):
    """Extracts a batch of features using your ResNet's backbone and neck."""
    feat = model.backbone(img_tensors)
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]
    feat = model.neck(feat)
    return feat.squeeze(-1).squeeze(-1) # Ensure we get a [B, C] tensor

# ✅ CHANGE 4: Load the ImageNet validation dataset
val_dataset = ImageFolder(root=imagenet_val_path, transform=resnet_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

all_preds = []
all_labels = []

with torch.no_grad():
    for img_tensors, labels in tqdm(val_loader, desc="Classifying ImageNet Images"):
        img_tensors = img_tensors.to(device)
        
        # 1. Extract image features
        image_features = extract_image_features(img_tensors)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 2. Calculate similarity against the 1000 synthesized embeddings
        similarity = image_features @ synthesized_embeddings.T
        
        # 3. Get predictions
        preds = similarity.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ---------------- CALCULATE AND PRINT FINAL ACCURACY ----------------
accuracy = accuracy_score(all_labels, all_preds)

print(f"\n✅ Zero-shot 'reconstruction' on ImageNet is complete!")
print(f"Accuracy: {accuracy * 100:.2f}%")
