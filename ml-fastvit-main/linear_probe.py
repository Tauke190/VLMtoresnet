import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from tqdm import tqdm

from timm.models import create_model, load_checkpoint

# ---- CONFIG ----
MODEL_NAME = "fastvit_sa36"  # Change as needed
MODEL_CKPT = "checkpoints/CLIPtoResNet/model_best_aircraft.pth.tar"  # <-- Set this!
NUM_CLASSES = 100  # CIFAR100
BATCH_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- DATA ----
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

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
            # Use forward_features if available, else fallback
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
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# ---- LINEAR PROBE ----
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, multi_class="multinomial", solver="lbfgs")
classifier.fit(train_features, train_labels)

# ---- EVALUATION ----
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")