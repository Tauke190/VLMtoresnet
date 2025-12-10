import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from timm.models import create_model, safe_model_name
from CLIP.dataloaders import aircraft as aircraft_dataloader


def build_aircraft_clip_text_features(clip_model, class_names, device, template_file):
    with open(template_file, "r") as f:
        templates = [line.strip() for line in f if line.strip()]

    all_text_features = []
    clip_model.eval()
    with torch.no_grad():
        for class_name in class_names:
            texts = [template.format(class_name) for template in templates]
            text_tokens = torch.cat([clip.tokenize(t) for t in texts]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            class_feature = text_features.mean(dim=0)
            class_feature = class_feature / class_feature.norm()
            all_text_features.append(class_feature)

    return torch.stack(all_text_features, dim=0)  # [num_classes, dim]


def setup_aircraft_loader(aircraft_root, device, template_file, num_workers=4, batch_size=64):
    if not aircraft_root or not os.path.isdir(aircraft_root):
        raise ValueError(f"Invalid aircraft_data_dir: {aircraft_root}")

    clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    dataset = aircraft_dataloader(
        root=aircraft_root,
        train=False,
        transform=preprocess,
    )

    class_names = getattr(dataset, "categories", None) or getattr(dataset, "classes", None)
    if class_names is None:
        raise RuntimeError("Aircraft dataset has no 'categories' or 'classes' attribute.")

    text_features = build_aircraft_clip_text_features(
        clip_model, class_names, device=device, template_file=template_file
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, text_features, class_names


def load_backbone(args, device):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        global_pool=args.gp if args.gp is not None else "avg",
    )

    ckpt = torch.load(args.model_checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    print(f"Loaded backbone {safe_model_name(args.model)} from {args.model_checkpoint}")
    return model


def load_projector(projector_ckpt_path, device):
    ckpt = torch.load(projector_ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    w = state["weight"]
    in_dim = w.shape[1]
    out_dim = w.shape[0]

    projector = nn.Linear(in_dim, out_dim)
    projector.load_state_dict(state, strict=True)
    projector.to(device)
    projector.eval()

    print(
        f"Loaded projector from {projector_ckpt_path} "
        f"(in_dim={in_dim}, out_dim={out_dim})"
    )
    return projector


def evaluate_aircraft_zeroshot(loader, text_features, model, projector, device):
    text_features = text_features.to(device)
    text_features = F.normalize(text_features.float(), dim=-1)

    correct_top1 = 0.0
    correct_top5 = 0.0
    total = 0

    backbone = model
    if hasattr(model, "module"):
        backbone = model.module

    backbone.eval()
    projector.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if hasattr(backbone, "forward_features"):
                feats = backbone.forward_features(images)
            else:
                feats = backbone(images)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4:
                feats = feats.mean(dim=[2, 3])

            feats = feats.float()
            feats = projector(feats)
            feats = F.normalize(feats, dim=-1)

            logits = 100.0 * feats @ text_features.T  # [B, num_classes]

            maxk = min(5, logits.size(1))
            _, pred = logits.topk(maxk, 1, True, True)  # [B, maxk]
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_top1 += correct[:1].reshape(-1).float().sum(0).item()
            correct_top5 += correct[:maxk].reshape(-1).float().sum(0).item()
            total += targets.size(0)

    top1 = 100.0 * correct_top1 / max(total, 1)
    top5 = 100.0 * correct_top5 / max(total, 1)
    return top1, top5


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot Aircraft eval (FastViT + projector + CLIP text)")
    parser.add_argument("--model", type=str, default="fastvit_sa36", help="Backbone model name")
    parser.add_argument("--num-classes", type=int, default=1000, help="Num classes of backbone head (unused here)")
    parser.add_argument("--gp", type=str, default=None, help="Global pool type (passed to timm.create_model)")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to backbone checkpoint (.pth.tar)")
    parser.add_argument("--projector-checkpoint", type=str, required=True, help="Path to projector checkpoint (.pth.tar)")
    parser.add_argument("--aircraft-data-dir", type=str, required=True, help="FGVC Aircraft dataset root")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--template-file",
        type=str,
        default=os.path.join("CLIP", "dataloaders", "templates", "fgvc_aircraft.txt"),
        help="Path to Aircraft prompt template file",
    )
    parser.add_argument("--device", type=str, default=None, help='cuda or cpu (default: auto)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader, text_features, class_names = setup_aircraft_loader(
        args.aircraft_data_dir,
        device=device,
        template_file=args.template_file,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )
    print(f"Aircraft dataset: {len(class_names)} classes, {len(loader.dataset)} images")

    model = load_backbone(args, device)
    projector = load_projector(args.projector_checkpoint, device)

    top1, top5 = evaluate_aircraft_zeroshot(loader, text_features, model, projector, device)

    print(f"\nZero-shot Aircraft accuracy:")
    print(f"  Top-1: {top1:.2f}%")
    print(f"  Top-5: {top5:.2f}%")


if __name__ == "__main__":
    main()