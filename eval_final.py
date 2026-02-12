import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import time
import logging
import models

from torchvision import transforms
from torchvision.datasets import ImageFolder

from CLIP.dataloaders.aircraft import aircraft as aircraft_dataloader
from CLIP.dataloaders.oxford_pets import OxfordPets
from CLIP.dataloaders.food101 import Food101
from CLIP.dataloaders.ucf101 import UCF101
from CLIP.dataloaders import DiffisionImages


# ---------------- Utils ----------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------- Backbone ----------------
def load_backbone(args, device):

    print("Creating model from local models package...")
    model_fn = getattr(models, args.model)
    model = model_fn(num_classes=args.num_classes)

    print("Loading checkpoint via torch.load()...")
    state = torch.load(args.model_checkpoint, map_location="cpu")

    if isinstance(state, dict):
        if "state_dict_ema" in state:
            state = state["state_dict_ema"]
        elif "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print("\n===== CHECKPOINT LOAD REPORT =====")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("==================================\n")

    model.to(device)
    model.eval()

    print(f"Backbone ready: {args.model}")
    return model


# ---------------- Data ----------------
def setup_loaders(dataset_name, dataset_root, batch_size=128, num_workers=4):

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

    if dataset_name == "aircraft":
        train_ds = aircraft_dataloader(root=dataset_root, train=True, transform=preprocess)
        test_ds = aircraft_dataloader(root=dataset_root, train=False, transform=preprocess)

    elif dataset_name == "oxfordpet":
        train_ds = OxfordPets(root=dataset_root, train=True, transform=preprocess)
        test_ds = OxfordPets(root=dataset_root, train=False, transform=preprocess)

    elif dataset_name == "food101":
        train_ds = Food101(root=dataset_root, train=True, transform=preprocess)
        test_ds = Food101(root=dataset_root, train=False, transform=preprocess)

    elif dataset_name == "imagenet":
        train_ds = ImageFolder(os.path.join(dataset_root, "train"), transform=preprocess)
        test_ds = ImageFolder(os.path.join(dataset_root, "validation"), transform=preprocess)

    elif dataset_name == "ucf101":
        train_ds = UCF101(root=dataset_root, train=True, transform=preprocess)
        test_ds = UCF101(root=dataset_root, train=False, transform=preprocess)

    elif dataset_name == "diffusion":
        train_ds = DiffisionImages(root=dataset_root, train=True, transform=preprocess)
        test_ds = DiffisionImages(root=dataset_root, train=False, transform=preprocess)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ---------------- Feature extraction ----------------
def extract_features(loader, backbone, device, mode="forward_features"):
    all_features, all_labels = [], []
    model = backbone.module if hasattr(backbone, "module") else backbone
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device)

            if mode == "backbone1" and hasattr(model, "forward_backbone"):
                feats = model.forward_backbone(images)
                B, C, H, W = feats.shape
                feats = feats.reshape(B, C, -1).mean(-1)

            elif mode == "classification_neck" and hasattr(model, "forward_classification_neck"):
                feats = model.forward_classification_neck(images)

            else:
                feats = model.forward_features(images) if hasattr(model, "forward_features") else model(images)

                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                if feats.ndim == 4:
                    feats = feats.mean(dim=[2, 3])

            feats = F.normalize(feats.float(), dim=-1)

            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


# ---------------- Classifier Evaluation ----------------
def evaluate_classifier(model, loader, device):
    model.eval()
    correct1, correct5, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Classifier Eval"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[1]

            _, pred1 = outputs.topk(1, dim=1)
            _, pred5 = outputs.topk(5, dim=1)

            correct1 += (pred1.squeeze() == labels).sum().item()

            for i in range(labels.size(0)):
                if labels[i] in pred5[i]:
                    correct5 += 1

            total += labels.size(0)

    print("\nClassifier Results")
    print(f"Top-1 Accuracy: {100*correct1/total:.3f}%")
    print(f"Top-5 Accuracy: {100*correct5/total:.3f}%")


# ---------------- Args ----------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-mode",
                        choices=["forward_features", "backbone1", "classification_neck", "classifier"],
                        default="forward_features")

    parser.add_argument("--model", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)

    parser.add_argument("--C", type=float, default=0.316)
    parser.add_argument("--max-iter", type=int, default=1000)

    return parser.parse_args()


# ---------------- Main ----------------
def main():
    start_time = time.time()

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Feature mode: {args.feature_mode}")
    print(f"Using device: {device}")

    train_loader, test_loader = setup_loaders(
        args.dataset, args.data_dir, args.batch_size, args.workers
    )

    backbone = load_backbone(args, device)

    total, trainable = count_parameters(backbone)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    if args.feature_mode == "classifier":
        evaluate_classifier(backbone, test_loader, device)
        return

    print("Extracting train features...")
    train_features, train_labels = extract_features(train_loader, backbone, device, args.feature_mode)

    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, backbone, device, args.feature_mode)

    print("Training logistic regression...")
    clf = LogisticRegression(
        random_state=0,
        C=args.C,
        max_iter=args.max_iter,
        n_jobs=-1,
        solver="lbfgs",
        multi_class="multinomial",
        verbose=1,
    )
    clf.fit(train_features, train_labels)

    preds = clf.predict(test_features)
    acc1 = (preds == test_labels).mean() * 100.0

    probs = clf.predict_proba(test_features)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    acc5 = np.mean([y in t for y, t in zip(test_labels, top5)]) * 100.0

    print(f"\nTop-1 Accuracy: {acc1:.3f}%")
    print(f"Top-5 Accuracy: {acc5:.3f}%")
    print(f"Runtime: {(time.time() - start_time)/60:.2f} min")


if __name__ == "__main__":
    main()
