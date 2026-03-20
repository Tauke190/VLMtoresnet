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
import clip

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

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------- Backbone ----------------
def load_backbone(args, device):

    print("Creating model from local models package...")
    model_fn = getattr(models, args.model)
    
    # Vanilla models don't have a projector; all others do
    if args.model in models.VANILLA_MODELS:
        model = model_fn(num_classes=args.num_classes)
    else:
        print("Detected projector model - using projector parameters")
        model = model_fn(
            num_classes=args.num_classes,
            freeze_backbone=False,  # Don't freeze for evaluation
            clip_dim=768,
            nonscalar_logit_scale=False
        )

    # Load pretrained backbone first if available
    pretrained_path = getattr(args, 'pretrained_backbone', None)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f" Loading pretrained backbone from: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(pretrained_state, dict):
            if "state_dict" in pretrained_state:
                pretrained_state = pretrained_state["state_dict"]
            elif "state_dict_ema" in pretrained_state:
                pretrained_state = pretrained_state["state_dict_ema"]
        
        # Clean pretrained state
        clean_pretrained = {}
        for k, v in pretrained_state.items():
            if k.startswith("module."):
                k = k[7:]
            # Only load backbone weights, skip head and projector
            if not any(x in k for x in ["head", "projector", "logit_scale"]):
                clean_pretrained[k] = v
        
        # Load pretrained backbone
        model.load_state_dict(clean_pretrained, strict=False)
        print(f" Loaded {len(clean_pretrained)} pretrained backbone weights")

    print("Loading checkpoint via torch.load()...")
    state = torch.load(args.model_checkpoint, map_location="cpu")

    if isinstance(state, dict):
        if "state_dict_ema" in state:
            state_dict = state["state_dict_ema"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model" in state:
            state_dict = state["model"]
    else:
        state_dict = state

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        clean_state[k] = v

    print(f"\n Analyzing checkpoint for frozen backbone training...")
    
    # Check if this is a frozen backbone checkpoint
    projector_keys = [k for k in clean_state.keys() if "projector" in k]
    backbone_keys = [k for k in clean_state.keys() if not any(x in k for x in ["projector", "head", "logit_scale"])]
    
    print(f"Projector keys: {len(projector_keys)}")
    print(f"Backbone keys: {len(backbone_keys)}")
    
    # For frozen backbone training, we might need to load pretrained backbone separately
    if len(projector_keys) > 0 and len(backbone_keys) == 0:
        print("  Detected frozen backbone training (only projector weights in checkpoint)")
        print(" You may need to load a pretrained backbone separately for best results")
        print(" Current evaluation will use randomly initialized backbone")
    
    # Handle num_classes mismatch for classification head
    if "head.weight" in clean_state and "head.bias" in clean_state:
        checkpoint_head_shape = clean_state["head.weight"].shape[0]
        model_head_shape = model.head.weight.shape[0]
        
        if checkpoint_head_shape != model_head_shape:
            print(f"  Num classes mismatch detected:")
            print(f"   Checkpoint: {checkpoint_head_shape} classes")
            print(f"   Model: {model_head_shape} classes")
            
            # Option 1: Remove head weights (current behavior)
            if not getattr(args, 'keep_original_head', False):
                print(f"   Removing classification head weights from checkpoint...")
                del clean_state["head.weight"]
                del clean_state["head.bias"]
                print(" Classification head weights removed - will use random initialization")
            else:
                print(f"   Keeping original head weights - will adapt to {model_head_shape} classes")
                # Adapt head weights to new number of classes
                old_head_weight = clean_state["head.weight"]
                old_head_bias = clean_state["head.bias"]
                
                if checkpoint_head_shape > model_head_shape:
                    # Truncate head weights
                    clean_state["head.weight"] = old_head_weight[:model_head_shape]
                    clean_state["head.bias"] = old_head_bias[:model_head_shape]
                    print(f" Head weights truncated from {checkpoint_head_shape} to {model_head_shape} classes")
                elif checkpoint_head_shape < model_head_shape:
                    # Pad head weights with random initialization
                    padding_weight = torch.randn(model_head_shape - checkpoint_head_shape, old_head_weight.shape[1])
                    padding_bias = torch.randn(model_head_shape - checkpoint_head_shape)
                    clean_state["head.weight"] = torch.cat([old_head_weight, padding_weight], dim=0)
                    clean_state["head.bias"] = torch.cat([old_head_bias, padding_bias], dim=0)
                    print(f" Head weights padded from {checkpoint_head_shape} to {model_head_shape} classes")

    # Load state dict and handle the return value
    load_result = model.load_state_dict(clean_state, strict=False)
    
    # When strict=False, load_state_dict returns None, so we need to check manually
    if load_result is None:
        # Manually check for missing and unexpected keys
        model_state_dict = model.state_dict()
        missing = [k for k in model_state_dict if k not in clean_state]
        unexpected = [k for k in clean_state if k not in model_state_dict]
    else:
        missing, unexpected = load_result

    print("\n===== CHECKPOINT LOAD REPORT =====")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if missing:
        # Check if only head keys are missing (expected when num_classes changed)
        head_only = all(k.startswith("head.") for k in missing)
        if head_only:
            print(f"Missing: {missing}   (expected: head was removed due to num_classes mismatch)")
        else:
            print("Missing:", missing[:5])  # Show first 5 missing keys
    if unexpected:
        print("Unexpected:", unexpected[:5])  # Show first 5 unexpected keys
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

            elif mode == "projector" and hasattr(model, "projector"):
                # Extract CLIP-space projector embeddings
                outputs = model(images)
                if isinstance(outputs, tuple):
                    # Projector models return (projected_embed, cls_out, features, logit_scale)
                    projected_embed = outputs[0]  # CLIP-space embeddings
                    feats = projected_embed
                else:
                    feats = outputs

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

def _load_diffusion_captions(dataset_root):
    """Load captions from the diffusion dataset caption files."""
    captions_2 = os.path.join(dataset_root, "caption_2k.txt")
    captions_5 = os.path.join(dataset_root, "caption_5k.txt")
    captions = []
    for path in (captions_2, captions_5):
        with open(path, "r") as f:
            captions.extend(line.strip() for line in f if line.strip())
    return captions


def _build_diffusion_text_features_fastvit_x(dataset_root, clip_model, device):
    """
    Build text features for the diffusion dataset using caption-per-sample
    matching (not class templates). Captions are encoded via the shared CLIP
    text encoder that was loaded alongside the FastViT-X projector checkpoint.

    Returns:
        text_features : Tensor [N, clip_dim], float32, L2-normalised
        captions       : list[str]  – the ordered caption list
    """
    all_captions = _load_diffusion_captions(dataset_root)

    # The DiffisionImages dataloader splits: first 6 000 → train, last 1 000 → test
    # We need the test slice for zero-shot evaluation
    test_captions = all_captions[-1000:]
    print(f" Diffusion captions (test): {len(test_captions)}")

    text_tokens = clip.tokenize(test_captions).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)

    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features, test_captions


def _build_template_text_features_fastvit_x(dataset_name, dataset_root, clip_model, device):
    """
    Build averaged template-based text features for standard classification
    datasets using the shared CLIP text encoder.
    """
    template_aliases = {"imagenet": "imagenet1k"}
    template_name = template_aliases.get(dataset_name, dataset_name)
    template_file = os.path.join(
        "CLIP", "dataloaders", "templates", f"{template_name}.txt"
    )

    if not os.path.exists(template_file):
        print(f"  Template file not found: {template_file}")
        templates = ["a photo of a {}."]
    else:
        with open(template_file, "r") as f:
            templates = [ln.strip() for ln in f if ln.strip()]

    # Resolve class names per dataset
    if dataset_name == "imagenet":
        classes_file = os.path.join(
            os.path.dirname(__file__), "misc", "imagenet_classes.txt"
        )
        if not os.path.exists(classes_file):
            raise FileNotFoundError(
                f"ImageNet class names file not found: {classes_file}"
            )
        with open(classes_file, "r") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]

    elif dataset_name == "food101":
        dataset = Food101(root=dataset_root, train=False, transform=None)
        class_names = (
            dataset.categories
            if hasattr(dataset, "categories")
            else getattr(dataset, "classes", [str(i) for i in range(101)])
        )

    elif dataset_name == "ucf101":
        ucf_file = os.path.join("CLIP", "dataloaders", "classes", "ucf101.txt")
        if os.path.exists(ucf_file):
            with open(ucf_file, "r") as f:
                class_names = [ln.strip() for ln in f if ln.strip()]
        else:
            class_names = [str(i) for i in range(101)]

    elif dataset_name == "aircraft":
        dataset = aircraft_dataloader(root=dataset_root, train=False, transform=None)
        class_names = getattr(dataset, "classes", [str(i) for i in range(100)])

    elif dataset_name == "oxfordpet":
        dataset = OxfordPets(root=dataset_root, train=False, transform=None)
        class_names = getattr(dataset, "classes", [str(i) for i in range(37)])

    else:
        raise ValueError(
            f"Unsupported dataset for template-based zero-shot: {dataset_name}"
        )

    print(f" Classes: {len(class_names)} | Templates: {len(templates)}")

    with torch.no_grad():
        text_features = []
        for classname in class_names:
            texts = [t.format(classname.replace("_", " ")) for t in templates]
            tokens = clip.tokenize(texts).to(device)
            cls_feats = clip_model.encode_text(tokens)
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
            cls_feats = cls_feats.mean(dim=0)
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
            text_features.append(cls_feats)

        text_features = torch.stack(text_features, dim=0).float()  # [C, D]

    return text_features, class_names


def setup_zeroshot_evaluation_fastvit_x(
    dataset_name, dataset_root, device, batch_size=128, num_workers=4
):
    print(f" [FastViT-X] Setting up zero-shot evaluation for {dataset_name}...")

    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    # CLIP normalisation pipeline (override clip_preprocess to ensure 224 crop)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    # ---------- text features ----------
    if dataset_name == "diffusion":
        text_features, class_info = _build_diffusion_text_features_fastvit_x(
            dataset_root, clip_model, device
        )
        test_dataset = DiffisionImages(root=dataset_root, train=False, transform=preprocess)
    else:
        text_features, class_info = _build_template_text_features_fastvit_x(
            dataset_name, dataset_root, clip_model, device
        )
        if dataset_name == "imagenet":
            test_dataset = ImageFolder(
                os.path.join(dataset_root, "validation"), transform=preprocess
            )
        elif dataset_name == "food101":
            test_dataset = Food101(root=dataset_root, train=False, transform=preprocess)
        elif dataset_name == "ucf101":
            test_dataset = UCF101(root=dataset_root, train=False, transform=preprocess)
        elif dataset_name == "aircraft":
            test_dataset = aircraft_dataloader(
                root=dataset_root, train=False, transform=preprocess
            )
        elif dataset_name == "oxfordpet":
            test_dataset = OxfordPets(root=dataset_root, train=False, transform=preprocess)
        else:
            raise ValueError(f"Unsupported dataset for FastViT-X zero-shot: {dataset_name}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_ctx = {
        "text_features": text_features,
        "class_info": class_info,
        "loader": test_loader,
        "dataset_name": dataset_name,
        "is_diffusion": dataset_name == "diffusion",
    }

    print(" [FastViT-X] Zero-shot evaluation setup complete")
    return eval_ctx


def run_zeroshot_evaluation_fastvit_x(eval_ctx, model, device):
    print(" [FastViT-X] Running zero-shot evaluation...")

    text_features = eval_ctx["text_features"].to(device)
    loader = eval_ctx["loader"]

    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="[FastViT-X] Zero-shot"):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            image_features = outputs[0] if isinstance(outputs, tuple) else outputs
            image_features = image_features / (
                image_features.norm(dim=-1, keepdim=True) + 1e-6
            )

            logits = 100.0 * image_features.float() @ text_features.T

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1_m.update(acc1.item(), images.size(0))
            top5_m.update(acc5.item(), images.size(0))

    print(" [FastViT-X] Zero-shot evaluation complete")
    return top1_m.avg, top5_m.avg


def _build_diffusion_text_features_clip(dataset_root, clip_model, device, train=False):
    all_captions = _load_diffusion_captions(dataset_root)
    # train slice: first 6 000; test slice: last 1 000
    captions_slice = all_captions[:6000] if train else all_captions[-1000:]
    print(f" Diffusion captions ({'train' if train else 'test'}): {len(captions_slice)}")

    text_tokens = clip.tokenize(captions_slice).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)

    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features, captions_slice


def _build_template_text_features_clip(dataset_name, dataset_root, clip_model, device):
    template_aliases = {"imagenet": "imagenet1k"}
    template_name = template_aliases.get(dataset_name, dataset_name)
    template_file = os.path.join(
        "CLIP", "dataloaders", "templates", f"{template_name}.txt"
    )

    if not os.path.exists(template_file):
        templates = ["a photo of a {}."]
    else:
        with open(template_file, "r") as f:
            templates = [ln.strip() for ln in f if ln.strip()]

    # Resolve class names (same logic as fastvit_x variant)
    if dataset_name == "imagenet":
        classes_file = os.path.join(
            os.path.dirname(__file__), "misc", "imagenet_classes.txt"
        )
        with open(classes_file, "r") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]

    elif dataset_name == "food101":
        dataset = Food101(root=dataset_root, train=False, transform=None)
        class_names = getattr(
            dataset, "categories", getattr(dataset, "classes", [str(i) for i in range(101)])
        )

    elif dataset_name == "ucf101":
        ucf_file = os.path.join("CLIP", "dataloaders", "classes", "ucf101.txt")
        if os.path.exists(ucf_file):
            with open(ucf_file, "r") as f:
                class_names = [ln.strip() for ln in f if ln.strip()]
        else:
            class_names = [str(i) for i in range(101)]

    elif dataset_name == "aircraft":
        dataset = aircraft_dataloader(root=dataset_root, train=False, transform=None)
        class_names = getattr(dataset, "classes", [str(i) for i in range(100)])

    elif dataset_name == "oxfordpet":
        dataset = OxfordPets(root=dataset_root, train=False, transform=None)
        class_names = getattr(dataset, "classes", [str(i) for i in range(37)])

    else:
        raise ValueError(
            f"Unsupported dataset for CLIP-only template zero-shot: {dataset_name}"
        )

    print(f" Classes: {len(class_names)} | Templates: {len(templates)}")

    with torch.no_grad():
        text_features = []
        for classname in class_names:
            texts = [t.format(classname.replace("_", " ")) for t in templates]
            tokens = clip.tokenize(texts).to(device)
            cls_feats = clip_model.encode_text(tokens)
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
            cls_feats = cls_feats.mean(dim=0)
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
            text_features.append(cls_feats)

        text_features = torch.stack(text_features, dim=0).float()

    return text_features, class_names


def setup_zeroshot_evaluation_clip(
    dataset_name, dataset_root, device, batch_size=128, num_workers=4
):
    print(f" [CLIP] Setting up zero-shot evaluation for {dataset_name}...")

    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    if dataset_name == "diffusion":
        text_features, class_info = _build_diffusion_text_features_clip(
            dataset_root, clip_model, device, train=False
        )
        test_dataset = DiffisionImages(root=dataset_root, train=False, transform=preprocess)
    else:
        text_features, class_info = _build_template_text_features_clip(
            dataset_name, dataset_root, clip_model, device
        )
        if dataset_name == "imagenet":
            test_dataset = ImageFolder(
                os.path.join(dataset_root, "validation"), transform=preprocess
            )
        elif dataset_name == "food101":
            test_dataset = Food101(root=dataset_root, train=False, transform=preprocess)
        elif dataset_name == "ucf101":
            test_dataset = UCF101(root=dataset_root, train=False, transform=preprocess)
        elif dataset_name == "aircraft":
            test_dataset = aircraft_dataloader(
                root=dataset_root, train=False, transform=preprocess
            )
        elif dataset_name == "oxfordpet":
            test_dataset = OxfordPets(root=dataset_root, train=False, transform=preprocess)
        else:
            raise ValueError(f"Unsupported dataset for CLIP zero-shot: {dataset_name}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_ctx = {
        "text_features": text_features,
        "class_info": class_info,
        "loader": test_loader,
        "clip_model": clip_model,   # kept in ctx so run_ can use encode_image
        "dataset_name": dataset_name,
        "is_diffusion": dataset_name == "diffusion",
    }

    print(" [CLIP] Zero-shot evaluation setup complete")
    return eval_ctx


def run_zeroshot_evaluation_clip(eval_ctx, device):
    print(" [CLIP] Running zero-shot evaluation...")

    clip_model   = eval_ctx["clip_model"].to(device)
    text_features = eval_ctx["text_features"].to(device)
    loader        = eval_ctx["loader"]

    top1_m = AverageMeter()
    top5_m = AverageMeter()

    clip_model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="[CLIP] Zero-shot"):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features.float(), dim=-1)

            logits = 100.0 * image_features @ text_features.T

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1_m.update(acc1.item(), images.size(0))
            top5_m.update(acc5.item(), images.size(0))

    print(" [CLIP] Zero-shot evaluation complete")
    return top1_m.avg, top5_m.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

    parser.add_argument(
        "--feature-mode",
        choices=["forward_features","backbone1","classification_neck","classifier","projector","zeroshot"],
        default="forward_features",
    )

    parser.add_argument("--model", required=True,
                        help="Model name. Use 'clip' for CLIP-only zero-shot (no checkpoint needed).")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of output classes. Auto-detected from the dataset if not provided.",
    )
    parser.add_argument("--model-checkpoint", default=None,
                        help="Path to model checkpoint. Not required when --model clip.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)

    parser.add_argument("--C", type=float, default=0.316)
    parser.add_argument("--max-iter", type=int, default=1000)

    parser.add_argument(
        "--pretrained-backbone",
        default=None,
        help="Path to pretrained backbone weights",
    )
    parser.add_argument(
        "--keep-original-head",
        action="store_true",
        help="Keep and adapt original head weights instead of random initialization",
    )

    return parser.parse_args()


# ---------------- Num-classes introspection ----------------

DATASET_NUM_CLASSES = {
    "food101": 101,
    "imagenet": 1000,
    "oxfordpet": 37,
    "aircraft": 100,
    "ucf101": 101,
    "diffusion": 7000,
}

# ---------------- Main ----------------

def main():
    start_time = time.time()

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect num_classes from the dataset's own .classes attribute
    if args.num_classes is None:
        if args.dataset in DATASET_NUM_CLASSES:
            args.num_classes = DATASET_NUM_CLASSES[args.dataset]
        else:
            raise ValueError(
                f"--num-classes is required for unknown dataset '{args.dataset}'. "
                f"Known datasets: {list(DATASET_NUM_CLASSES.keys())}"
            )

    print(f"Feature mode : {args.feature_mode}")
    print(f"Model        : {args.model}")
    print(f"Using device : {device}")


    if args.feature_mode == "zeroshot":

        if args.model == "clip":
            print(" Routing to CLIP-only zero-shot evaluation")
            eval_ctx = setup_zeroshot_evaluation_clip(
                args.dataset, args.data_dir, device,
                args.batch_size, args.workers,
            )
            acc1, acc5 = run_zeroshot_evaluation_clip(eval_ctx, device)
        else:
            eval_ctx = setup_zeroshot_evaluation_fastvit_x(
                args.dataset, args.data_dir, device,
                args.batch_size, args.workers,
            )
            backbone = load_backbone(args, device)
            total, trainable = count_parameters(backbone)
            print(f"Total params: {total:,} | Trainable: {trainable:,}")
            acc1, acc5 = run_zeroshot_evaluation_fastvit_x(eval_ctx, backbone, device)

        print(f"\n Zero-Shot Evaluation Results")
        print(f"Top-1 Accuracy: {acc1:.3f}%")
        print(f"Top-5 Accuracy: {acc5:.3f}%")
        print(f"Runtime: {(time.time() - start_time)/60:.2f} min")
        return
    
    train_loader, test_loader = setup_loaders(
        args.dataset, args.data_dir, args.batch_size, args.workers
    )

    backbone = load_backbone(args, device)
    total, trainable = count_parameters(backbone)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    if args.feature_mode == "projector":
        print(" Using CLIP-space projector embeddings for linear probing")
        if hasattr(backbone, "projector"):
            proj = backbone.projector
            out_dim = (
                proj.out_features
                if hasattr(proj, "out_features")
                else getattr(getattr(proj, "fc2", None), "out_features", 768)
            )
            print(f" Projector found: CLIP dim = {out_dim}")
        else:
            print(" No projector found in model!")

    if args.feature_mode == "classifier":
        evaluate_classifier(backbone, test_loader, device)
        return

    # Linear probing
    print("Extracting train features...")
    train_features, train_labels = extract_features(
        train_loader, backbone, device, args.feature_mode
    )

    print("Extracting test features...")
    test_features, test_labels = extract_features(
        test_loader, backbone, device, args.feature_mode
    )

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
    acc1  = (preds == test_labels).mean() * 100.0

    probs = clf.predict_proba(test_features)
    top5  = np.argsort(probs, axis=1)[:, -5:]
    acc5  = np.mean([y in t for y, t in zip(test_labels, top5)]) * 100.0

    print(f"\n Linear Probe Evaluation Results")
    print(f"Top-1 Accuracy: {acc1:.3f}%")
    print(f"Top-5 Accuracy: {acc5:.3f}%")
    print(f"Runtime: {(time.time() - start_time)/60:.2f} min")


if __name__ == "__main__":
    main()
