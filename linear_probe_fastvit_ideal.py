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

from timm.models import create_model, safe_model_name, load_checkpoint
from torchvision import transforms

from CLIP.dataloaders.aircraft import aircraft as aircraft_dataloader
from CLIP.dataloaders.oxford_pets import OxfordPets
from CLIP.dataloaders.food101 import Food101


# ---------------- Utils ----------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------- Backbone ----------------
def load_backbone(args, device):
    model = create_model(
        args.model,
        pretrained=False,         
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
    )

    ckpt_path = args.model_checkpoint
    try:
        load_checkpoint(model, ckpt_path, use_ema=False)
        print("Loaded checkpoint using timm.load_checkpoint()")
    except Exception as e:
        print(f"timm.load_checkpoint failed ({e}), falling back to torch.load")
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint using torch.load()")

    model.to(device)
    model.eval()
    print(f"Backbone ready: {safe_model_name(args.model)}")
    return model


# ---------------- Data ----------------
def setup_linearprobe_loaders(dataset_name, dataset_root, batch_size=128, num_workers=4):
    """
    ONLY CHANGE HERE:
    - Standard ImageNet resize + normalize
    """

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

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

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, class_names


# ---------------- Feature extraction ----------------
def extract_features(loader, backbone, projector, device, use_projector=True):
    all_features = []
    all_labels = []

    model = backbone
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    if projector is not None:
        projector.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)

            if hasattr(model, "forward_features"):
                feats = model.forward_features(images)
            else:
                feats = model(images)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4:
                feats = feats.mean(dim=[2, 3])

            feats = feats.float()

            if use_projector and projector is not None:
                feats = projector(feats)

            # L2 normalize 
            feats = F.normalize(feats, dim=-1)

            all_features.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels


# ---------------- Args ----------------
def parse_args():
    parser = argparse.ArgumentParser()
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
    logging.info(f"Using device: {device}")
    logging.info(f"Args: {vars(args)}")

    train_loader, test_loader, class_names = setup_linearprobe_loaders(
        args.dataset, args.data_dir, args.batch_size, args.workers
    )

    logging.info(
        f"{args.dataset}: {len(class_names)} classes | "
        f"{len(train_loader.dataset)} train | "
        f"{len(test_loader.dataset)} test"
    )

    backbone = load_backbone(args, device)
    total, trainable = count_parameters(backbone)
    logging.info(f"Total params: {total:,} | Trainable: {trainable:,}")

    logging.info("Extracting train features...")
    train_features, train_labels = extract_features(
        train_loader, backbone, None, device
    )

    logging.info("Extracting test features...")
    test_features, test_labels = extract_features(
        test_loader, backbone, None, device
    )

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

    preds = clf.predict(test_features)
    acc1 = (preds == test_labels).mean() * 100.0

    probs = clf.predict_proba(test_features)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    acc5 = np.mean([y in t for y, t in zip(test_labels, top5)]) * 100.0

    logging.info(f"Top-1 Accuracy: {acc1:.3f}%")
    logging.info(f"Top-5 Accuracy: {acc5:.3f}%")
    logging.info(f"Total runtime: {(time.time() - start_time)/60:.2f} min")


if __name__ == "__main__":
    main()
