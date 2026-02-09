import argparse
import os

import clip
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import time
import logging
import models

from timm.models import create_model, safe_model_name
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
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
    )

    print("Loading checkpoint via torch.load()...")

    state = torch.load(args.model_checkpoint, map_location="cpu")

    # unwrap common checkpoint formats
    if isinstance(state, dict):

        if "state_dict_ema" in state:
            print("Using EMA weights")
            state = state["state_dict_ema"]

        elif "state_dict" in state:
            state = state["state_dict"]

        elif "model" in state:
            state = state["model"]

    # remove DDP prefix
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=False)

    model.to(device)
    model.eval()

    print(f"Backbone ready: {safe_model_name(args.model)}")

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
        class_names = train_ds.categories

    elif dataset_name == "oxfordpet":
        train_ds = OxfordPets(root=dataset_root, train=True, transform=preprocess)
        test_ds = OxfordPets(root=dataset_root, train=False, transform=preprocess)
        class_names = train_ds.categories

    elif dataset_name == "food101":
        train_ds = Food101(root=dataset_root, train=True, transform=preprocess)
        test_ds = Food101(root=dataset_root, train=False, transform=preprocess)
        class_names = train_ds.categories

    elif dataset_name == "imagenet":
        train_dir = os.path.join(dataset_root, "train")
        val_dir = os.path.join(dataset_root, "validation")  
        train_ds = ImageFolder(train_dir, transform=preprocess)
        test_ds = ImageFolder(val_dir, transform=preprocess)
        class_names = train_ds.classes

    elif dataset_name == "ucf101":
        train_ds = UCF101(root=dataset_root, train=True, transform=preprocess)
        test_ds = UCF101(root=dataset_root, train=False, transform=preprocess)
        class_names = list(range(101))

    elif dataset_name == "diffusion":
        train_ds = DiffisionImages(root=dataset_root, train=True, transform=preprocess)
        test_ds = DiffisionImages(root=dataset_root, train=False, transform=preprocess)

        if hasattr(train_ds, "labels"):
            class_names = list(set(train_ds.labels))
        else:
            class_names = []

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, class_names


# ---------------- Feature extraction ----------------
def extract_features(loader, backbone, device, mode="forward_features"):
    all_features, all_labels = [], []

    model = backbone.module if hasattr(backbone, "module") else backbone
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)

            if mode == "backbone1" and hasattr(model, "forward_backbone"):
                feats = model.forward_backbone(images)
                B, C, H, W = feats.shape
                feats = feats.reshape(B, C, -1).mean(-1)

            elif mode == "classification_neck" and hasattr(model, "forward_classification_neck"):
                feats = model.forward_classification_neck(images)

            elif mode == "classifier":
                feats = model(images)

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


# ---------------- Args ----------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-mode", choices=["linear", "logits"], default="linear")
    parser.add_argument(
        "--feature-mode",
        choices=["forward_features", "backbone1", "classification_neck", "classifier"],
        default="forward_features",
    )

    parser.add_argument("--model", default="fastvit_sa36")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--gp", default=None)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)

    parser.add_argument("--C", type=float, default=0.316)
    parser.add_argument("--max-iter", type=int, default=1000)

    return parser.parse_args()


# ---------------- Logging ----------------
def setup_logger(log_file="out.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


# ---------------- Main ----------------
def main():
    setup_logger("out.log")
    start_time = time.time()

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Eval mode: {args.eval_mode}")
    logging.info(f"Feature mode: {args.feature_mode}")
    logging.info(f"Using device: {device}")

    train_loader, test_loader, class_names = setup_loaders(
        args.dataset, args.data_dir, args.batch_size, args.workers
    )

    backbone = load_backbone(args, device)
    total, trainable = count_parameters(backbone)
    logging.info(f"Total params: {total:,} | Trainable: {trainable:,}")

    logging.info("Extracting train features...")
    train_features, train_labels = extract_features(train_loader, backbone, device, args.feature_mode)

    logging.info("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, backbone, device, args.feature_mode)

    logging.info("Training logistic regression...")
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

    if args.eval_mode == "linear":
        preds = clf.predict(test_features)
        acc1 = (preds == test_labels).mean() * 100.0

        probs = clf.predict_proba(test_features)
        top5 = np.argsort(probs, axis=1)[:, -5:]
        acc5 = np.mean([y in t for y, t in zip(test_labels, top5)]) * 100.0

        logging.info(f"Top-1 Accuracy: {acc1:.3f}%")
        logging.info(f"Top-5 Accuracy: {acc5:.3f}%")

    else:
        logits = clf.decision_function(test_features)

        acc1 = (np.argmax(logits, axis=1) == test_labels).mean() * 100.0
        top5 = np.argsort(logits, axis=1)[:, -5:]
        acc5 = np.mean([y in t for y, t in zip(test_labels, top5)]) * 100.0

        preds = clf.predict(test_features)
        assert np.all(np.argmax(logits, axis=1) == preds), "Logits sanity check failed"

        logging.info(f"Top-1 Accuracy (logits): {acc1:.3f}%")
        logging.info(f"Top-5 Accuracy (logits): {acc5:.3f}%")

    logging.info(f"Total runtime: {(time.time() - start_time)/60:.2f} min")


if __name__ == "__main__":
    main()
