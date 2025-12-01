import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_


class FastViT_lr(nn.Module):
    def __init__(
        self,
        base_model,
        lock: bool = True,
        token_std: float = 0.02,
    ):
        super().__init__()
        self.base = base_model
        self.is_locked = lock

        # Freeze backbone weights
        if lock:
            for p in self.base.parameters():
                p.requires_grad = False

        # SA36 @ 224 layout: (num_blocks, embed_dim, H, W)
        stage_configs = [
            (6, 64, 56, 56),  # Stage 1
            (6, 128, 28, 28),  # Stage 2
            (18, 256, 14, 14),  # Stage 3
            (6, 512, 7, 7),  # Stage 4
        ]

        self.blocks_spatial_tokens = nn.ModuleList()

        for num_blocks, embed_dim, H, W in stage_configs:
            stage_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, embed_dim, H, W))
                    for _ in range(num_blocks)
                ]
            )
            for t in stage_tokens:
                trunc_normal_(t, std=token_std)

            self.blocks_spatial_tokens.append(stage_tokens)

        # Attach token to each block before forward
        for s_idx, stage in enumerate(self.base.stages):
            blocks = getattr(stage, "blocks", None)
            if blocks is None:
                continue

            for b_idx, blk in enumerate(blocks):
                if getattr(blk, "_has_token_wrap", False):
                    continue

                orig_fwd = blk.forward

                def fwd_with_token(x, *args, orig_fwd=orig_fwd, s=s_idx, b=b_idx, **kwargs):
                    x = self.add_spatial_token(x, s, b)
                    return orig_fwd(x, *args, **kwargs)

                blk.forward = fwd_with_token
                blk._has_token_wrap = True

    def add_spatial_token(self, x, stage_idx, block_idx):
        token = self.blocks_spatial_tokens[stage_idx][block_idx]  # (1, C, H, W)

        return x + token

    def forward_features(self, x):
        return self.base.forward_features(x)

    def forward(self, x):
        return self.base(x)
