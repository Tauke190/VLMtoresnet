import torch
import json
from pathlib import Path
import clip

def zeroshot_classifier(classnames, templates, model, show_progress=True):
    """Creating zero-shot classifier weights (CLIP-style)."""
    with torch.no_grad():
        device = next(model.parameters()).device
        zeroshot_weights = []
        iterator = classnames
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(classnames, desc="Building zero-shot weights", total=len(classnames))
            except Exception:
                iterator = classnames
        for classname in iterator:
            texts = [template.format(classname) for template in templates]
            tokens = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(tokens).float()
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device=device, dtype=torch.float32)
    return zeroshot_weights


def get_teacher_features(model, images):
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(backbone, images):
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    # Ensure (B, D) shape for all timm backbones
    return pooled_features.flatten(1)

def compute_flops(model, resolution=(3, 224, 224)):
    from thop import profile
    model_cpu = model.to('cpu')
    dummy = torch.randn(1, *resolution)
    with torch.no_grad():
        flops, params = profile(model_cpu, inputs=(dummy,), verbose=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n***** FLOP TOTAL: {flops / 10 ** 9:.2f} GFLOPs *****")
    print(f"***** Model Parameters: {params:,} *****\n")
    return flops / 10 ** 9, params

def load_prompts_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            templates = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(templates)} templates from {filepath}.")
        return templates
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}.")
        return []

# Compute FLOPs on CPU to save VRAM
def compute_flops(model, resolution=(3, 224, 224)):
    try:
        from thop import profile
    except Exception:
        print("thop not installed; skipping FLOPs computation.")
        params = sum(p.numel() for p in model.parameters())
        print(f"***** Model Parameters: {params:,} *****")
        return None, params

    orig_device = next(model.parameters()).device
    model_cpu = model.to('cpu')
    dummy = torch.randn(1, *resolution)
    with torch.no_grad():
        flops, params = profile(model_cpu, inputs=(dummy,), verbose=False)
    model.to(orig_device)
    print(f"\n\n***** FLOP TOTAL: {flops / 10 ** 9:.2f} GFLOPs *****")
    print(f"***** Model Parameters: {params:,} *****\n")
    return flops / 10 ** 9, params

def load_imagenet_classnames(json_path="imagenet_class_index.json"):
    with open(json_path, "r") as f:
        idx_to_data = json.load(f)
    # Returns list indexed by class idx: human-readable name
    return [idx_to_data[str(i)][1].replace('_', ' ') for i in range(len(idx_to_data))]

# New: map synset -> readable name, then align to ImageFolder class order
def load_imagenet_synset_to_name(json_path="imagenet_class_index.json"):
    with open(json_path, "r") as f:
        idx_to_data = json.load(f)
    # idx -> (synset, name)
    return {idx_to_data[str(i)][0]: idx_to_data[str(i)][1].replace('_', ' ') for i in range(len(idx_to_data))}

def imagenet_aligned_classnames(dataset, json_path="imagenet_class_index.json"):
    syn_to_name = load_imagenet_synset_to_name(json_path)
    return [syn_to_name.get(syn, syn) for syn in dataset.classes]

def imagefolder_human_names(dataset):
    return [c.replace('_', ' ') for c in dataset.classes]

def read_txt(file_location):
    with open(file_location, 'r') as file:
        # Strip whitespace and drop empty lines robustly
        return [line.strip() for line in file if line.strip()]

def save_checkpoint(backbone, projector, epoch, project_root, script_path):
    checkpoint_dir = project_root / "distilled_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting by including epoch in filename
    checkpoint_path = checkpoint_dir / f"{Path(script_path).stem}.pt"
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")