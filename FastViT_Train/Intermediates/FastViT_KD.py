import torch
import torch.nn as nn
import torch.nn.functional as F
from FastViT_LR import FastViT_lr


class FastViT_CLIP(nn.Module):
    """
    FastViT_lr with projector(s) for CLIP distillation.
    Supports returning a 14x14 intermediate (stage2) + final.
    """

    def __init__(self, base_model, embed_dim=768, lock=True):
        super().__init__()

        self.fastvit = FastViT_lr(base_model, lock=lock)

        # Final feature dim for SA36 final map (after timm forward_features)
        self.feature_dim = 1024
        self.projector = nn.Linear(self.feature_dim, embed_dim)
        nn.init.trunc_normal_(self.projector.weight, std=0.001)
        nn.init.zeros_(self.projector.bias)

        # Stage2 (14x14) dim in SA36 is 256 channels
        self.stage2_dim = 256
        self.stage2_projector = nn.Linear(self.stage2_dim, embed_dim)
        nn.init.trunc_normal_(self.stage2_projector.weight, std=0.001)
        nn.init.zeros_(self.stage2_projector.bias)

        print("FastViT_CLIP created")
        print(f"  Final:  {self.feature_dim} → {embed_dim}")
        print(f"  Stage2: {self.stage2_dim} → {embed_dim}  (14×14)")

    def forward(self, x, return_intermediate=False):
        if return_intermediate:
            # Get real pipeline final + intermediates from LR
            final_map, feats = self.fastvit.forward_features(
                x, return_intermediates=True, stage_indices=[2]
            )
            stage2_map = feats["stage2"]  # 14×14 feature map

            # Pool + project stage2
            stage2_pooled = F.adaptive_avg_pool2d(stage2_map, 1).flatten(1)  # (B, 256)
            stage2_proj = self.stage2_projector(stage2_pooled).float()

            # Pool + project final
            final_pooled = F.adaptive_avg_pool2d(final_map, 1).flatten(1)  # (B, 1024)
            final_proj = self.projector(final_pooled).float()

            return final_proj, stage2_proj

        # Final-only path (unchanged behavior)
        final_map = self.fastvit.forward_features(x)
        final_pooled = F.adaptive_avg_pool2d(final_map, 1).flatten(1)
        final_proj = self.projector(final_pooled).float()
        return final_proj


def create_fastvit_clip(
    model_name="fastvit_sa36", pretrained=True, embed_dim=768, lock=True
):
    from timm import create_model

    base_model = create_model(model_name, pretrained=pretrained)
    return FastViT_CLIP(base_model, embed_dim=embed_dim, lock=lock)


if __name__ == "__main__":
    model = create_fastvit_clip("fastvit_sa36", pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    out = model(x)
    print("Final only:", out.shape)

    out_final, out_stage2 = model(x, return_intermediate=True)
    print("Final + stage2:", out_final.shape, out_stage2.shape)
