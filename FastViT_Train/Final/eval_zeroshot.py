import os
import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import clip

from timm.utils import reduce_tensor
from FastViT_KD import create_fastvit_clip

USE_EMA = True
BATCH_SIZE = 256
NUM_WORKERS = 8
TEMPERATURE = 100.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP preprocessing
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGENET_CLASSES_TXT = os.path.join(_BASE_DIR, "imagenet_classes.txt")
IMAGENET_LABELS_TXT = os.path.join(_BASE_DIR, "imagenet_labels.txt")


def _load_lines(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _create_prompts(class_names, templates):
    # Turn "goldfish" -> ["a photo of a goldfish", ...]
    class_names = [c.replace("_", " ") for c in class_names]
    return [[t.format(c) for t in templates] for c in class_names]


@torch.no_grad()
def _encode_text_prompts(clip_model, prompts, device):
    """
    clip_model: CLIP model (text encoder)
    prompts: list[list[str]] where prompts[i] is all templates for class i
    """
    text_features = []
    for class_prompts in tqdm(prompts, desc="Encode text"):
        tokens = clip.tokenize(class_prompts).to(device)
        emb = clip_model.encode_text(tokens)  # [T, D]
        emb = emb / emb.norm(dim=-1, keepdim=True)  # per-template norm
        emb = emb.mean(dim=0)  # average templates
        emb = emb / emb.norm()  # per-class norm
        text_features.append(emb)

    text_features = torch.stack(text_features, dim=0)  # [C, D]
    return text_features.float().to(device)


@torch.no_grad()
def prepare_zeroshot_head(
    clip_model=None,
    device_override=None,
    templates=None,
):
    """
    Build CLIP text features for ImageNet-1K using imagenet_classes.txt.

    Args:
        clip_model: optional CLIP model. If None, loads ViT-L/14.
        device_override: device string (e.g. "cuda:0"). If None, infer from model.
        templates: list of string templates. If None, uses ["a photo of a {}"].

    Returns:
        text_features: [num_classes, D] tensor
        class_names: list of strings, same ordering as rows in text_features
    """
    # Load CLIP if not provided
    if clip_model is None:
        clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
        clip_model.eval()

    if device_override is None:
        device_used = next(clip_model.parameters()).device
    else:
        device_used = device_override

    # Load classes / labels (labels only for logging / sanity)
    class_names = _load_lines(IMAGENET_CLASSES_TXT)
    if os.path.exists(IMAGENET_CLASSES_TXT):
        print(f"Loading ImageNet labels from {IMAGENET_CLASSES_TXT}...")

    try:
        imagenet_labels = _load_lines(IMAGENET_LABELS_TXT)
        if os.path.exists(IMAGENET_LABELS_TXT):
            print(f"Loading ImageNet labels from {IMAGENET_LABELS_TXT}...")
        if len(imagenet_labels) != len(class_names):
            print(
                f"[warn] imagenet_labels.txt has {len(imagenet_labels)} lines, "
                f"imagenet_classes.txt has {len(class_names)} lines."
            )
    except FileNotFoundError:
        imagenet_labels = None

    if templates is None:
        templates = ["a photo of a {}"]

    prompts = _create_prompts(class_names, templates)
    print(f"Encoding CLIP text features for {len(class_names)} classes...")
    text_features = _encode_text_prompts(clip_model, prompts, device_used)
    return text_features, class_names


@torch.no_grad()
def evaluate_zero_shot(
    model,
    loader,
    text_features,
    args=None,
    amp_autocast=None,
    temperature=TEMPERATURE,
):
    """
    Streaming zero-shot evaluation on a DataLoader.

    Args:
        model: student (FastViT) model that outputs feature vectors.
        loader: DataLoader yielding (input, target).
        text_features: [num_classes, D] CLIP text embeddings.
        args: optional argparse-like object from train.py
              (used for prefetcher, channels_last, distributed, world_size, local_rank).
        amp_autocast: context manager for AMP (e.g. torch.cuda.amp.autocast).
                      If None, uses a no-op context.
        temperature: logit scaling factor.

    Returns:
        OrderedDict({"zs_top1": top1, "zs_top5": top5}) with percentages.
    """
    from contextlib import suppress

    model.eval()
    device_used = next(model.parameters()).device
    text_features = text_features.to(device=device_used, dtype=torch.float32)

    if amp_autocast is None:
        amp_autocast = suppress

    prefetcher = getattr(args, "prefetcher", False) if args is not None else False
    channels_last = getattr(args, "channels_last", False) if args is not None else False
    distributed = getattr(args, "distributed", False) if args is not None else False
    world_size = getattr(args, "world_size", 1) if args is not None else 1
    local_rank = getattr(args, "local_rank", 0) if args is not None else 0

    top1_correct = 0.0
    top5_correct = 0.0
    total = 0.0

    for input, target in tqdm(loader, desc="Zero-shot eval"):
        if not prefetcher:
            input = input.to(device_used, non_blocking=True)
            target = target.to(device_used, non_blocking=True)

        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            feats = model(input)  # [B, D]
            feats = feats.float()
            feats = F.normalize(feats, dim=-1)  # unit norm

            logits = temperature * (feats @ text_features.t())  # [B, C]

        # Top-1
        pred1 = logits.argmax(dim=-1)
        top1_correct += (pred1 == target).sum().item()

        # Top-5
        top5 = logits.topk(5, dim=-1).indices
        top5_correct += (top5 == target.unsqueeze(1)).any(dim=1).sum().item()

        total += target.size(0)

    stats = torch.tensor(
        [top1_correct, top5_correct, total],
        device=device_used,
        dtype=torch.float32,
    )

    if distributed and world_size > 1:
        stats = reduce_tensor(stats, world_size)

    top1 = (stats[0] / stats[2] * 100.0).item()
    top5 = (stats[1] / stats[2] * 100.0).item()

    if local_rank == 0:
        print(f"[Zero-shot] Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")

    return OrderedDict([("zs_top1", top1), ("zs_top5", top5)])


def _build_val_loader(val_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def _load_fastvit_from_checkpoint(checkpoint_path, device_used):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if USE_EMA and "state_dict_ema" in ckpt:
            state_dict = ckpt["state_dict_ema"]
        else:
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt

    # Strip common prefixes
    for prefix in ["module.", "model.", "student."]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}

    # Handle spatial_tokens → blocks_spatial_tokens rename
    new_state_dict = {}
    for k, v in state_dict.items():
        if "fastvit.spatial_tokens" in k:
            k = k.replace("fastvit.spatial_tokens", "fastvit.blocks_spatial_tokens")
        elif "spatial_tokens" in k and "blocks_spatial_tokens" not in k:
            k = k.replace("spatial_tokens", "blocks_spatial_tokens")
        new_state_dict[k] = v
    state_dict = new_state_dict

    model = create_fastvit_clip(
        model_name="fastvit_sa36",
        pretrained=False,
        embed_dim=768,
        lock=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device_used).eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot eval for FastViT checkpoints"
    )
    parser.add_argument(
        "-c",
        "--checkpoints",
        nargs="+",
        default=["model_best.pth.tar"],
        help="Path(s) to checkpoint .pth.tar files",
    )
    parser.add_argument(
        "--val-dir",
        default="/datasets/ImageNet2012nonpub/validation",
        help="ImageNet validation directory (ImageFolder layout)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
    )
    args = parser.parse_args()

    print(f"Loading ImageNet val from: {args.val_dir}")
    dataset, val_loader = _build_val_loader(
        args.val_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Validation set: {len(dataset)} images, {len(dataset.classes)} classes")

    # Build CLIP text head (loads CLIP internally)
    text_features, class_names = prepare_zeroshot_head()
    print(f"Prepared zeroshot head for {len(class_names)} classes")

    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path} (not found)")
            continue

        print(f"\n{'='*60}\nEvaluating {ckpt_path}\n{'='*60}")
        model = _load_fastvit_from_checkpoint(ckpt_path, device)

        metrics = evaluate_zero_shot(model, val_loader, text_features)
        print(f"\nZero-shot Top-1: {metrics['zs_top1']:.2f}%")
        print(f"Zero-shot Top-5: {metrics['zs_top5']:.2f}%\n")


if __name__ == "__main__":
    main()
