import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

import models
from timm.models import create_model, load_checkpoint

# ---- CONFIG ----
MODEL_NAME = "fastvit_sa36"  # Change as needed
MODEL_CKPT = "/checkpoints/CLIPtoResNet/fastvit_sa36/model_best.pth.tar"  # Path to your backbone checkpoint
NUM_CLASSES = 1000  # ImageNet
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_ROOT = "/home/c3-0/datasets/ImageNet"  # <-- Set this to your ImageNet root

# ---- DATA ----
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dir = os.path.join(IMAGENET_ROOT, "train")
val_dir = os.path.join(IMAGENET_ROOT, "validation")

train_dataset = ImageFolder(train_dir, transform=preprocess)
val_dataset = ImageFolder(val_dir, transform=preprocess)

# ---- MODEL ----
model = create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    in_chans=3,
    global_pool=None,
)
load_checkpoint(model, MODEL_CKPT, use_ema=False)
model.to(DEVICE)
model.eval()

def get_features(dataset):
    all_features = []
    all_labels = []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            if hasattr(model, "forward_features"):
                feats = model.forward_features(images)
            else:
                feats = model(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4:
                feats = feats.mean(dim=[2, 3])
            all_features.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

# ---- FEATURE EXTRACTION ----
print("Extracting train features...")
train_features, train_labels = get_features(train_dataset)
print("Extracting val features...")
val_features, val_labels = get_features(val_dataset)

# ---- LINEAR PROBE ----
print("Training linear classifier...")
classifier = LogisticRegression(
    random_state=0, C=0.316, max_iter=1000, verbose=1,
    multi_class="multinomial", solver="lbfgs"
)
classifier.fit(train_features, train_labels)

# ---- EVALUATION ----
print("Evaluating...")
predictions = classifier.predict(val_features)
accuracy = np.mean((val_labels == predictions).astype(float)) * 100.
print(f"Top-1 Accuracy = {accuracy:.3f}")

# Optionally, compute Top-5 accuracy
probs = classifier.predict_proba(val_features)
top5 = np.argsort(probs, axis=1)[:, -5:]
top5_acc = np.mean([label in top5_row for label, top5_row in zip(val_labels, top5)]) * 100.
print(f"Top-5 Accuracy = {top5_acc:.3f}")