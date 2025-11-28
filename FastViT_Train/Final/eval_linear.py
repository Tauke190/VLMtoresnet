import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from FastViT_KD import create_fastvit_clip

USE_EMA = True
BATCH_SIZE = 256
NUM_WORKERS = 8

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


def load_fastvit_from_checkpoint(checkpoint_path):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if USE_EMA and "state_dict_ema" in ckpt:
            state_dict = ckpt["state_dict_ema"]
            which = "state_dict_ema"
        else:
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            which = "state_dict/model"
    else:
        state_dict = ckpt
        which = "raw"

    raw_keys = list(state_dict.keys())
    print(f"Using {which} | keys={len(raw_keys)}")
    print("Raw keys sample:", raw_keys[:5])

    # Strip ONLY wrapper/DDP prefixes
    for prefix in ["module.", "model.", "student."]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}

    # Safe token remap (only touch OLD names)
    new_state_dict = {}
    for k, v in state_dict.items():
        if "fastvit.spatial_tokens" in k:
            k = k.replace("fastvit.spatial_tokens", "fastvit.blocks_spatial_tokens")
        elif "spatial_tokens" in k and "blocks_spatial_tokens" not in k:
            k = k.replace("spatial_tokens", "blocks_spatial_tokens")
        new_state_dict[k] = v
    state_dict = new_state_dict

    clean_keys = list(state_dict.keys())
    print("Clean keys sample:", clean_keys[:5])

    model = create_fastvit_clip(
        model_name="fastvit_sa36",
        pretrained=False,
        embed_dim=768,
        lock=False,
    )

    msg = model.load_state_dict(state_dict, strict=False)
    print(
        f"Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}"
    )
    if msg.missing_keys:
        print("  Missing sample:", msg.missing_keys[:5])
    if msg.unexpected_keys:
        print("  Unexpected sample:", msg.unexpected_keys[:5])

    model.to(device).eval()
    return model


def get_features(model, dataset):
    feats_list, labels_list = [], []

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extract"):
            feats = model(images.to(device))  # [B, 768]
            feats = feats.flatten(1)
            feats_list.append(feats.cpu())
            labels_list.append(labels.cpu())

    feats = torch.cat(feats_list).numpy()
    labels = torch.cat(labels_list).numpy()
    return feats, labels


checkpoints = [
    "model_best.pth.tar",
    "checkpoint-37.pth.tar",
    # "./CheckPoint30/checkpoint-30.pth.tar",
    # "./CheckPoint30/Wcheckpoint-30.pth.tar",
]

for checkpoint_path in checkpoints:
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {checkpoint_path} (not found)")
        continue

    print(f"\n{'='*50}\n{checkpoint_path}\n{'='*50}")

    model = load_fastvit_from_checkpoint(checkpoint_path)

    print("Extracting train features...")
    train_features, train_labels = get_features(model, train_dataset)

    print("Extracting val features...")
    val_features, val_labels = get_features(model, val_dataset)
    print("Feature extraction completed.")

    print("Training logistic regression...")
    classifier = LogisticRegression(
        random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1
    )
    classifier.fit(train_features, train_labels)

    print("Evaluating...")
    predictions = classifier.predict(val_features)
    top1_accuracy = np.mean((val_labels == predictions).astype(float)) * 100.0

    probas = classifier.predict_proba(val_features)
    top5_preds = np.argsort(probas, axis=1)[:, -5:]
    top5_accuracy = (
        np.mean([label in top5_preds[i] for i, label in enumerate(val_labels)]) * 100.0
    )

    print(f"\nTop-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
