import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from FastViT_Train.Final.FastViT_KD import create_fastvit_clip

USE_EMA = True
BATCH_SIZE = 256
NUM_WORKERS = 8
TEMPERATURE = 100.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP preprocessing
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
)

val_dataset = datasets.ImageFolder(
    root="/datasets/ImageNet2012nonpub/validation", transform=transform
)


def load_lines(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def create_prompts(classes, templates):
    classes = [c.replace("_", " ") for c in classes]
    return [[t.format(c) for t in templates] for c in classes]


def load_fastvit_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if USE_EMA and "state_dict_ema" in ckpt:
            state_dict = ckpt["state_dict_ema"]
        else:
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt

    for prefix in ["module.", "model.", "student."]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}

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
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_text_prompts(clip_model, prompts):
    text_features = []
    for class_prompts in tqdm(prompts, desc="Encode text"):
        tokens = clip.tokenize(class_prompts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        emb = emb / emb.norm()
        text_features.append(emb)
    text_features = torch.stack(text_features, dim=0)
    return text_features.float()


@torch.no_grad()
def get_image_features(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    feats_list, labels_list = [], []
    for images, labels in tqdm(loader, desc="Encode images"):
        images = images.to(device, non_blocking=True)
        feats = model(images)
        feats_list.append(feats.float())
        labels_list.append(labels.to(device))
    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return feats, labels


def evaluate_zeroshot(image_features, text_features, labels):
    # both float32 now
    logits = TEMPERATURE * (image_features @ text_features.T)
    top1 = (logits.argmax(dim=-1) == labels).float().mean().item() * 100.0
    top5 = (logits.topk(5, dim=-1).indices == labels.unsqueeze(1)).any(
        dim=1
    ).float().mean().item() * 100.0
    return top1, top5


def main():
    classes = load_lines("imagenet_classes.txt")
    templates = load_lines("imagenet_templates.txt")
    prompts = create_prompts(classes, templates)

    print(
        f"Loaded validation: {len(val_dataset)} images, {len(val_dataset.classes)} classes"
    )

    print("Loading CLIP ViT-L/14 text encoder...")
    clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    print("Encoding text prompts...")
    text_features = encode_text_prompts(clip_model, prompts).to(device)

    checkpoints = [
        "checkpoint-37.pth.tar",
        "model_best.pth.tar",
    ]
    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path} (not found)")
            continue

        print(f"\n{'='*60}\nEvaluating {ckpt_path}\n{'='*60}")
        model = load_fastvit_from_checkpoint(ckpt_path)

        print("Encoding images...")
        image_features, labels = get_image_features(model, val_dataset)

        top1, top5 = evaluate_zeroshot(image_features, text_features, labels)
        print(f"\nZero-shot Top-1: {top1:.2f}%")
        print(f"Zero-shot Top-5: {top5:.2f}%\n")


if __name__ == "__main__":
    main()
