import torch
import torch.nn as nn
import torch.nn.functional as F
from FastViT_LR import FastViT_lr


class FastViT_CLIP(nn.Module):
    """
    FastViT_lr with projector for CLIP distillation
    """

    def __init__(self, base_model, embed_dim=768, lock=True):
        """
        Args:
            base_model: FastViT model (from timm)
            embed_dim: CLIP embedding dimension (768 for L/14)
            lock: Freeze backbone
        """
        super().__init__()

        # FastViT_lr (frozen backbone + trainable tokens)
        self.fastvit = FastViT_lr(base_model, lock=lock)

        # Get feature dimension (1024 for FastViT SA36)
        self.feature_dim = 1024

        # Simple linear projector: 1024 → 768
        self.projector = nn.Linear(self.feature_dim, embed_dim)

        # Change std=0.02 to std=0.001 (Tiny weights)
        nn.init.trunc_normal_(self.projector.weight, std=0.001)
        nn.init.zeros_(self.projector.bias)

        print(f"✓ FastViT_CLIP created")
        print(f"  Feature dim: {self.feature_dim} → CLIP dim: {embed_dim}")

    def forward(self, x, return_features=False):
        # FastViT features (Let this stay in float16 for speed)
        features = self.fastvit.forward_features(x)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        projected = self.projector(features)

        # 1. Cast to float32 (Safe Mode) just for this calculation
        projected = projected.float()

        # 2. Normalize with safety epsilon
        projected = F.normalize(projected, dim=-1, eps=1e-6)

        if return_features:
            logits = self.fastvit.head(features)
            return logits, projected

        return projected


def create_fastvit_clip(
    model_name="fastvit_sa36", pretrained=True, embed_dim=768, lock=True
):
    """
    Helper function to create FastViT_CLIP

    Args:
        model_name: FastViT variant
        pretrained: Load pretrained weights
        embed_dim: CLIP embedding dimension (768 for L/14)
        lock: Freeze backbone

    Returns:
        FastViT_CLIP model
    """
    from timm import create_model

    base_model = create_model(model_name, pretrained=pretrained)
    model = FastViT_CLIP(base_model, embed_dim=embed_dim, lock=lock)

    return model


# Test
if __name__ == "__main__":
    model = create_fastvit_clip("fastvit_sa36", pretrained=False)

    # Check parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

    # Test forward
    x = torch.randn(2, 3, 224, 224)
    features = model(x)
    print(f"\nOutput features: {features.shape}")  # (2, 768)
    print("✓ Model ready!")
