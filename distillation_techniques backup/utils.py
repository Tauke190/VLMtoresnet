import torch
import json
from pathlib import Path

def get_teacher_features(model, images):
    with torch.no_grad():
        features = model.encode_image(images)
    return features

def get_student_features(backbone, images):
    feature_map = backbone.forward_features(images)
    pooled_features = backbone.global_pool(feature_map)
    return pooled_features

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
        content = file.read(); content = str(content); content = content.split('\n', -1)
    try: content.remove("")
    except: pass
    return content

def save_checkpoint(backbone, projector, epoch, project_root, script_path):
    checkpoint_dir = project_root / "distilled_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / (Path(script_path).stem + ".pt")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")