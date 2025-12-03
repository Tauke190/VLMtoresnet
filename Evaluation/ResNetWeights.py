import os
import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import OrderedDict

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_dataset = datasets.ImageFolder(
    root="/datasets/ImageNet2012nonpub/train", transform=transform
)
val_dataset = datasets.ImageFolder(
    root="/datasets/ImageNet2012nonpub/validation", transform=transform
)


def load_resnet50_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint type: {type(checkpoint)}")

    # Extract state_dict
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {checkpoint.keys()}")
        if "backbone_state_dict" in checkpoint:
            state_dict = checkpoint["backbone_state_dict"]
            print("Using 'backbone_state_dict'")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Using 'state_dict'")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Using 'model'")
        else:
            state_dict = checkpoint
            print("Using checkpoint as state_dict")
    else:
        state_dict = checkpoint
        print("Checkpoint is OrderedDict")

    # Strip 'module.' prefix if present
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        print("Stripping 'module.' prefix")
        state_dict = OrderedDict(
            [(k.replace("module.", "", 1), v) for k, v in state_dict.items()]
        )

    print(f"State dict has {len(state_dict)} keys")
    print(f"First key: {next(iter(state_dict.keys()))}")

    # Load into ResNet50 and convert to feature extractor
    model = models.resnet50(weights=None)
    model.load_state_dict(state_dict, strict=False)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    print("Model loaded successfully\n")

    return model


def get_features(model, dataset):
    all_features = []
    all_labels = []

    loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    with torch.no_grad():
        for images, labels in tqdm(loader):
            feats = model(images.to(device))
            feats = feats.flatten(1)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    return (
        torch.cat(all_features).numpy(),
        torch.cat(all_labels).numpy(),
    )


# Checkpoints to evaluate
checkpoints = [
    "finalfeature_distillation_cliptoresnet.pt",
    "contrastive_distillation_cliptoresnet.pt",
    "intermediate_feature_distillation_cliptoresnet.pt",
    "masked_generative_distillation.pt",
]

for checkpoint_path in checkpoints:
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {checkpoint_path} (not found)")
        continue

    print(f"\n{'='*50}\n{checkpoint_path}\n{'='*50}")

    # Load model
    model = load_resnet50_from_checkpoint(checkpoint_path)
    print("Extracting features...")

    # Extract features
    train_features, train_labels = get_features(model, train_dataset)
    val_features, val_labels = get_features(model, val_dataset)
    print("Feature extraction completed.")

    # Train and evaluate
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    print("Classifier trained. Evaluating...")

    # Top-1 accuracy
    predictions = classifier.predict(val_features)
    top1_accuracy = np.mean((val_labels == predictions).astype(float)) * 100.0

    # Top-5 accuracy
    probas = classifier.predict_proba(val_features)
    top5_preds = np.argsort(probas, axis=1)[:, -5:]
    top5_accuracy = (
        np.mean([label in top5_preds[i] for i, label in enumerate(val_labels)]) * 100.0
    )

    print(f"\nTop-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
