import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from mmengine.config import Config
from mmcls.models import build_classifier
from mmengine.runner import load_checkpoint
from PIL import Image
from torchvision import transforms

# ---------------- CONFIGURATION ----------------
# --- Your Custom ResNet Model ---
config_path = 'configs/imagenet/resnet50_8xb32_in1k_strong_aug_coslr_300.py'
checkpoint_path = 'resnet50_scalekd_e300_new.pth'

# --- ImageNet Validation Path ---
imagenet_val_path = '~/data/datasets/ImageNet/val'

# --- Script Settings ---
BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- STEP 1: LOAD MODEL AND EXTRACT CLASS EMBEDDINGS ----------------

# Load your complete, trained ImageClassifier model
print("Loading custom ResNet-50D Image Classifier...")
cfg = Config.fromfile(config_path)
model = build_classifier(cfg.model)
load_checkpoint(model, checkpoint_path, map_location=device)
model.to(device)
model.eval()

# Extract the learned class representations from the classification head's weight matrix
print("Extracting learned class embeddings from the model's head...")
with torch.no_grad():
    # The weights of the final fully-connected layer are our class embeddings
    class_embeddings = model.head.fc.weight.clone() # Shape: [1000, 2048]
    # Normalize them for cosine similarity calculation
    class_embeddings = F.normalize(class_embeddings, p=2, dim=1)


# Define the image transforms your model was trained with
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
    feat = model.neck(feat) # Shape: [batch_size, 2048, 1, 1]
    return feat.squeeze() # Shape: [batch_size, 2048]

# ---------------- STEP 2: EVALUATE ON THE VALIDATION SET ----------------
print("\nEvaluating on the ImageNet validation set...")

# Create the validation dataset and dataloader
val_dataset = ImageFolder(root=imagenet_val_path, transform=resnet_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

top1_correct = 0
top5_correct = 0

with torch.no_grad():
    for img_tensors, labels in tqdm(val_loader, desc="Classifying ImageNet"):
        img_tensors = img_tensors.to(device)
        labels = labels.to(device)
        
        # 1. Extract image features with your ResNet
        image_features = extract_image_features(img_tensors) # Shape: [batch_size, 2048]
        
        # 2. Normalize the image features for cosine similarity
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 3. Calculate similarity against all 1000 class embeddings
        # This is equivalent to a matrix multiplication
        similarity = image_features @ class_embeddings.T # Shape: [batch_size, 1000]
        
        # 4. Get Top-5 predictions
        _, top5_preds = torch.topk(similarity, 5, dim=-1)
        
        # 5. Check correctness
        top1_preds = top5_preds[:, 0]
        top1_correct += (top1_preds == labels).sum().item()
        top5_correct += (top5_preds == labels.view(-1, 1)).any(dim=1).sum().item()

# ---------------- CALCULATE AND PRINT FINAL ACCURACY ----------------
total_samples = len(val_dataset)
top1_accuracy = (top1_correct / total_samples) * 100
top5_accuracy = (top5_correct / total_samples) * 100

print(f"\n✅ Classification with your checkpoint is complete!")
print(f"Total Validation Samples: {total_samples}")
print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
