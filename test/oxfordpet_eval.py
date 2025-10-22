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
# Point this to the validation/test folder of the Oxford-IIIT Pet dataset
oxford_pet_val_path = '~/data/datasets/oxford_pet/val'
# You still need the ImageNet class list for the knowledge transfer
imagenet_class_index_path = 'imagenet_class_index.json'

# --- Script Settings ---
BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- STEP 1: LOAD MODELS AND EXTRACT IMAGENET KNOWLEDGE ----------------

# Load your complete, trained ImageClassifier model
print("Loading custom ResNet-50D Image Classifier...")
cfg = Config.fromfile(config_path)
model = build_classifier(cfg.model)
load_checkpoint(model, checkpoint_path, map_location=device)
model.to(device)
model.eval()

# Extract the learned class representations (knowledge) from the head's weight matrix
print("Extracting learned ImageNet embeddings from the model's head...")
with torch.no_grad():
    imagenet_knowledge_vectors = model.head.fc.weight.clone().float() # Shape: [1000, 2048]

# ---------------- STEP 2: BUILD THE TEXT-BASED KNOWLEDGE BRIDGE ----------------

print("Loading CLIP and building text-based knowledge bridge...")
clip_model, _ = clip.load("ViT-B/32", device=device)

# Define the source (ImageNet) and target (Oxford Pet) class names
with open(imagenet_class_index_path, 'r') as f:
    class_idx = json.load(f)
imagenet_classes = [class_idx[str(k)][1].replace("_", " ") for k in sorted([int(i) for i in class_idx.keys()])]

oxford_pet_classes = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
    'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
    'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees',
    'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher',
    'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue',
    'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
    'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]

# Create text features for both sets of classes
with torch.no_grad():
    # ImageNet text features
    imagenet_text_features = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
    imagenet_text_features = clip_model.encode_text(imagenet_text_features)
    imagenet_text_features = F.normalize(imagenet_text_features, p=2, dim=1)
    
    # Oxford Pet text features
    pet_text_features = clip.tokenize([f"a photo of a {c}" for c in oxford_pet_classes]).to(device)
    pet_text_features = clip_model.encode_text(pet_text_features)
    pet_text_features = F.normalize(pet_text_features, p=2, dim=1)

# Calculate the similarity matrix (the bridge) between pet and ImageNet classes
# Shape: [37, 1000]
text_similarity_bridge = pet_text_features @ imagenet_text_features.T

# ---------------- STEP 3: SYNTHESIZE PET CLASS EMBEDDINGS ----------------

print("Synthesizing new pet breed embeddings...")
# Use the bridge to transfer knowledge from the ImageNet vectors to the pet vectors
# [37, 1000] @ [1000, 2048] -> [37, 2048]
synthesized_pet_embeddings = text_similarity_bridge.float() @ imagenet_knowledge_vectors
# Normalize the newly created embeddings for accurate comparison
synthesized_pet_embeddings = F.normalize(synthesized_pet_embeddings, p=2, dim=1)


# ---------------- STEP 4: EVALUATE ON THE OXFORD PET DATASET ----------------
print("\nEvaluating on the Oxford-IIIT Pet validation set...")
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
    return feat.squeeze()

val_dataset = ImageFolder(root=oxford_pet_val_path, transform=resnet_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

all_preds = []
all_labels = []

with torch.no_grad():
    for img_tensors, labels in tqdm(val_loader, desc="Classifying Pet Images"):
        img_tensors = img_tensors.to(device)
        
        # 1. Extract image features
        image_features = extract_image_features(img_tensors)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 2. Calculate similarity against the 37 synthesized pet embeddings
        similarity = image_features @ synthesized_pet_embeddings.T
        
        # 3. Get predictions
        preds = similarity.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ---------------- CALCULATE AND PRINT FINAL ACCURACY ----------------
accuracy = accuracy_score(all_labels, all_preds)

print(f"\n✅ Zero-shot classification on Oxford-IIIT Pet is complete!")
print(f"Accuracy: {accuracy * 100:.2f}%")
