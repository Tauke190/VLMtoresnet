import torch
import clip

try:
    # When used as a package
    from .utils import get_student_features
except Exception:
    # Fallback if run as a flat script
    from utils import get_student_features

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

def evaluate_zero_shot(backbone, projector, loader, zs_weights, device=None):
    backbone.eval()
    projector.eval()
    if device is None:
        device = next(backbone.parameters()).device
    zs_weights = zs_weights.to(device=device, dtype=torch.float32)

    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            student_feats = get_student_features(backbone, images)
            proj_feats = projector(student_feats).float()
            proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)
            logits = 100.0 * (proj_feats @ zs_weights)  # [B, C]
            _, top5 = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5[:, 0] == labels).sum().item()
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5