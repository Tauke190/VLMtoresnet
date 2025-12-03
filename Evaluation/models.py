import torch
from torchvision import transforms
from typing import Callable, Tuple


def load_clip_vitl14(
    device: str,
) -> Tuple[torch.nn.Module, transforms.Compose, Callable]:
    """Load CLIP ViT-L/14"""
    import clip

    print("Loading CLIP ViT-L/14...")
    model, base_preprocess = clip.load("ViT-L/14", device=device)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            base_preprocess,
        ]
    )
    model.eval()

    def extract_fn(images):
        return model.encode_image(images)

    return model, transform, extract_fn


def load_eva02_clip(
    device: str,
) -> Tuple[torch.nn.Module, transforms.Compose, Callable]:
    """Load EVA02-CLIP-L/14"""
    import open_clip

    print("Loading EVA02-CLIP-L/14...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
    )
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            preprocess_val,
        ]
    )
    model.to(device)
    model.eval()

    def extract_fn(images):
        return model.encode_image(images)

    return model, transform, extract_fn


def load_fastvit_sa36(
    device: str,
) -> Tuple[torch.nn.Module, transforms.Compose, Callable]:
    """Load FastViT SA36"""
    import timm

    print("Loading FastViT SA36...")
    model = timm.create_model("fastvit_sa36", pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    base_transform = timm.data.create_transform(**data_config, is_training=False)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            base_transform,
        ]
    )

    def extract_fn(images):
        features = model(images)
        if isinstance(features, (list, tuple)):
            features = features[0]
        if len(features.shape) > 2:
            features = features.squeeze()
        return features

    return model, transform, extract_fn


def load_convmixer_768_32(
    device: str,
) -> Tuple[torch.nn.Module, transforms.Compose, Callable]:
    """Load ConvMixer 768/32"""
    import timm

    print("Loading ConvMixer 768/32...")
    model = timm.create_model("convmixer_768_32", pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_fn(images):
        features = model(images)
        if len(features.shape) > 2:
            features = features.squeeze()
        return features

    return model, transform, extract_fn


def load_scalekd_resnet50(
    device: str, checkpoint_path: str
) -> Tuple[torch.nn.Module, transforms.Compose, Callable]:
    """Load ScaleKD ResNet-50-D"""
    from mmpretrain.registry import MODELS
    from typing import Dict

    print("Loading ScaleKD ResNet-50-D...")

    # Build model
    cfg = dict(
        type="ImageClassifier",
        backbone=dict(
            type="ResNet",
            depth=50,
            style="pytorch",
            deep_stem=True,
            avg_down=True,
            norm_cfg=dict(type="BN"),
        ),
        neck=dict(type="GlobalAveragePooling"),
        head=dict(
            type="LinearClsHead",
            num_classes=1000,
            in_channels=2048,
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
            topk=(1, 5),
        ),
    )
    model = MODELS.build(cfg)

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state: Dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)

    if any(k.startswith("student.") for k in state.keys()):
        raise RuntimeError(
            "Checkpoint contains 'student.' keys. "
            "Please run ScaleKD's pth_transfer.py first."
        )

    if any(k.startswith(("backbone.", "head.")) for k in state.keys()):
        filtered = {k: v for k, v in state.items() if "num_batches_tracked" not in k}
        model.load_state_dict(filtered, strict=False)
    else:
        new_state: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if "num_batches_tracked" in k:
                continue
            if k.startswith("fc."):
                new_state["head." + k] = v
            else:
                new_state["backbone." + k] = v
        model.load_state_dict(new_state, strict=False)

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_fn(images):
        feats = model.backbone(images)
        feats = model.neck(feats)
        if isinstance(feats, tuple):
            feats = feats[0]
        if len(feats.shape) > 2:
            feats = feats.squeeze()
        return feats

    return model, transform, extract_fn


# Model registry
MODEL_REGISTRY = {
    "clip_vitl14": load_clip_vitl14,
    "eva02_clip": load_eva02_clip,
    "fastvit_sa36": load_fastvit_sa36,
    "convmixer_768_32": load_convmixer_768_32,
    "scalekd_resnet50": load_scalekd_resnet50,
}


def get_model(model_name: str, device: str, checkpoint_path: str = None):
    """
    Get model, transform, and feature extraction function.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    loader_fn = MODEL_REGISTRY[model_name]

    if model_name == "scalekd_resnet50":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for scalekd_resnet50")
        return loader_fn(device, checkpoint_path)
    else:
        return loader_fn(device)
