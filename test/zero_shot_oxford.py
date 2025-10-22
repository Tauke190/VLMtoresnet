import os
import torch
import numpy as np
import clip
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from mmengine.config import Config
from mmcls.models import build_classifier
from mmengine.runner import load_checkpoint
from PIL import Image # <--- 1. IMPORT THE IMAGE LIBRARY

# ---------------- CONFIGURATION ----------------
config_path = 'configs/imagenet/resnet50_8xb32_in1k_strong_aug_coslr_300.py'
checkpoint_path = 'resnet50_scalekd_e300_new.pth'
dataset_root_path = '~/data/datasets/oxford_pet/val'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- LOAD YOUR IMAGE ENCODER (ResNet) ----------------
# (This part is unchanged)
cfg = Config.fromfile(config_path)
image_encoder = build_classifier(cfg.model)
load_checkpoint(image_encoder, checkpoint_path, map_location=device)
image_encoder.to(device)
image_encoder.eval()

def extract_image_feature(model, img_tensor):
    with torch.no_grad():
        feat = model.backbone(img_tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        feat = model.neck(feat)
    return feat

# ---------------- LOAD TEXT ENCODER (CLIP) ----------------
# (This part is unchanged)
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# ---------------- DEFINE AND ENCODE TEXT LABELS ----------------
# (This part is unchanged)
class_names = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
    'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
    'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees',
    'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher',
    'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue',
    'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
    'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]

print("Encoding text labels...")
with torch.no_grad():
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    text_features = clip_model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)


# ---------------- DATASET & PREPROCESS ----------------
# Define the transforms for your ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create the dataset but we will get paths from it, not tensors
dataset = ImageFolder(root=dataset_root_path)


# ---------------- ZERO-SHOT CLASSIFICATION LOOP ----------------
all_preds = []
all_labels = []

# <--- 2. MODIFY THE LOOP
# Instead of `for img, label in dataset`, we get the path and label

for path, label in tqdm(dataset.samples, desc="Classifying images"):
    # Open the raw image with PIL
    img = Image.open(path)

    # --- THIS IS THE FIX ---
    # Ensure the image is in RGB format, stripping any alpha channel
    img = img.convert('RGB')

    # A. Process the image for your ResNet
    img_tensor_resnet = resnet_transform(img).unsqueeze(0).to(device)
    
    # B. Process the image for CLIP
    image_for_clip = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        #image_feature_resnet = extract_image_feature(image_encoder,img_tensor_resnet)
        #image_feature_resnet = image_feature_resnet.view(image_feature_resnet[0],-1) # flattens the image features
        #image_feature_resnet /= image_feature_resnet.norm(dim=-1, keepdim=True)

        image_feature_clip = clip_model.encode_image(image_for_clip)
        image_feature_clip /= image_feature_clip.norm(dim=-1, keepdim=True)


    # C. Find the best match
    similarity = (100.0 * image_feature_clip @ text_features.T).softmax(dim=-1)
    pred = similarity.argmax().cpu().item()

    all_preds.append(pred)
    all_labels.append(label)

# ---------------- CALCULATE AND PRINT ACCURACY ----------------
# (The rest of the script is the same)
accuracy = accuracy_score(all_labels, all_preds)

print(f"? Zero-shot classification complete!")
print(f"Accuracy: {accuracy * 100:.2f}%")
