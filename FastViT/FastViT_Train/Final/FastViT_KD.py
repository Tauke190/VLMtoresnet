import torch
import torch.nn as nn
import torch.nn.functional as F
from FastViT_LR import FastViT_lr


class FastViT_CLIP(nn.Module):
    """
    FastViT_lr with projector for CLIP distillation
    """

    def __init__(self, base_model, embed_dim=768, lock=True):
        super().__init__()

        # FastViT_lr (frozen backbone + trainable tokens)
        self.fastvit = FastViT_lr(base_model, lock=lock)

        # Get feature dimension (1024 for FastViT SA36)
        self.feature_dim = 1024

        # Simple linear projector: 1024 to 768
        self.projector = nn.Linear(self.feature_dim, embed_dim)

        nn.init.trunc_normal_(self.projector.weight, std=0.02)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        features = self.fastvit.forward_features(x)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        projected = self.projector(features)
        return projected.float()


def create_fastvit_clip(
    model_name="fastvit_sa36", pretrained=True, embed_dim=768, lock=True
):
    from timm import create_model

    base_model = create_model(model_name, pretrained=pretrained)
    model = FastViT_CLIP(base_model, embed_dim=embed_dim, lock=lock)

    return model
