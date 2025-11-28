import os
import torch
import clip

from DatasetLoader import (
    get_stanford_cars_loaders,
    get_food101_loaders,
    get_aircraft_loaders,
    get_gtsrb_loaders,
    get_fer2013_loaders,
    get_country211_loaders,
    get_ucf101_loaders,
    get_sst2_loaders,
)

# Basic config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-L/14@336px"
DATA_ROOT = "./data"
TEMPLATE_DIR = "./templates"
CLASSES_DIR = "./classes"
BATCH_SIZE = 128
NUM_WORKERS = 8

# Datasets to evaluate
DATASETS = [
    dict(
        name="fgvc_aircraft",
        loader_fn=get_aircraft_loaders,
        template_name="fgvc_aircraft.txt",
        classes_name="fgvc_aircraft.txt",
    ),
    dict(
        name="gtsrb",
        loader_fn=get_gtsrb_loaders,
        template_name="GTSRB.txt",
        classes_name="gtsrb.txt",
    ),
    dict(
        name="food101",
        loader_fn=get_food101_loaders,
        template_name="food101.txt",
        classes_name="food101.txt",
    ),
    dict(
        name="fer2013",
        loader_fn=get_fer2013_loaders,
        template_name="fer2013.txt",
        classes_name="fer2013.txt",
    ),
    dict(
        name="country211",
        loader_fn=get_country211_loaders,
        template_name="country211.txt",
        classes_name="country211.txt",
    ),
    dict(
        name="ucf101",
        loader_fn=get_ucf101_loaders,
        template_name="ucf101.txt",
        classes_name="ucf101.txt",
    ),
    dict(
        name="cars",
        loader_fn=get_stanford_cars_loaders,
        template_name="cars.txt",
        classes_name="cars.txt",
    ),
    dict(
        name="sst2",
        loader_fn=get_sst2_loaders,
        template_name="sst2.txt",
        classes_name="sst2.txt",
    ),
]


def _read_txt_lines(root_dir, name, required):
    """Load non-empty, non-comment lines from root_dir/name(.txt)."""
    path = os.path.join(root_dir, name)
    if not path.endswith(".txt"):
        path_txt = path + ".txt"
        if os.path.isfile(path_txt):
            path = path_txt

    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(f"File not found for '{name}' in {root_dir}")
        return None

    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def load_templates(name):
    """Load prompt templates for zero-shot classification."""
    return _read_txt_lines(TEMPLATE_DIR, name, required=True)


def load_classnames(name):
    """Load class names from CLASSES_DIR, if the file exists."""
    return _read_txt_lines(CLASSES_DIR, name, required=False)


def get_classnames_from_dataset(dataset):
    """Infer class names from a dataset object."""
    # torchvision-style metadata
    if hasattr(dataset, "classes"):
        return list(dataset.classes)

    if hasattr(dataset, "class_to_idx"):
        idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]

    # Fallback: infer from label arrays
    labels = None
    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            labels = getattr(dataset, attr)
            break

    if labels is not None:
        if isinstance(labels, torch.Tensor):
            uniq = labels.unique().cpu().tolist()
        else:
            uniq = sorted({int(x) for x in labels})
        uniq = sorted(uniq)
        return [str(i) for i in uniq]

    # Nothing usable
    raise ValueError(
        "Dataset has no .classes/.class_to_idx and no targets/labels/y; "
        "please provide a classes file and set classes_name."
    )


@torch.no_grad()
def build_zeroshot_classifier(model, classnames, templates, device):
    """Build CLIP text embeddings for each class using prompt templates."""
    weights = []
    for cls in classnames:
        texts = [t.format(cls) for t in templates]
        tokens = clip.tokenize(texts).to(device)

        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        emb = feats.mean(dim=0)
        emb = emb / emb.norm()
        weights.append(emb)

    return torch.stack(weights, dim=0)  # (C, d)


@torch.no_grad()
def eval_dataset(
    name,
    loader_fn,
    template_name,
    model,
    preprocess,
    classes_name=None,
    device=DEVICE,
):
    """Evaluate zero-shot CLIP on a single dataset."""
    print(f"\n=== {name} ===")

    # DatasetLoader must return (train_loader, test_loader)
    train_loader, test_loader = loader_fn(
        DATA_ROOT, preprocess, BATCH_SIZE, NUM_WORKERS
    )

    # 1) Try classes file
    classnames = None
    if classes_name:
        classnames = load_classnames(classes_name)
        if classnames:
            print(f"Classes from: {os.path.join(CLASSES_DIR, classes_name)}")
        else:
            print(
                f"Warning: class file {classes_name} not found, "
                "trying dataset metadata."
            )

    # 2) Fallback to dataset metadata
    if classnames is None:
        try:
            classnames = get_classnames_from_dataset(train_loader.dataset)
            print("Classes from dataset metadata")
        except ValueError as e:
            raise ValueError(
                f"{name}: no class file found and dataset has no class metadata. "
                f"Create ./classes/{name}.txt or fix DatasetLoader."
            ) from e


    templates = load_templates(template_name)
    print(f"Num classes: {len(classnames)}, templates: {len(templates)}")

    weights = build_zeroshot_classifier(model, classnames, templates, device)
    num_classes = weights.shape[0]
    use_top5 = num_classes >= 5

    correct1 = 0
    correct5 = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feats @ weights.T  # (B, C)

        pred1 = logits.argmax(dim=-1)
        correct1 += (pred1 == labels).sum().item()

        if use_top5:
            top5 = logits.topk(5, dim=-1).indices
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    top1 = 100.0 * correct1 / total
    if use_top5:
        top5 = 100.0 * correct5 / total
        print(f"Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    else:
        print(f"Top-1: {top1:.2f}%")

    return top1


def main():
    """Run zero-shot evaluation over all configured datasets."""
    print(f"Using CLIP model: {MODEL_NAME} on {DEVICE}")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    for cfg in DATASETS:
        eval_dataset(
            name=cfg["name"],
            loader_fn=cfg["loader_fn"],
            template_name=cfg["template_name"],
            model=model,
            preprocess=preprocess,
            classes_name=cfg["classes_name"],
            device=DEVICE,
        )


if __name__ == "__main__":
    main()
