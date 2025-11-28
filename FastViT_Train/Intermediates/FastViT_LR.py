import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class FastViT_lr(nn.Module):
    """
    FastViT with per-block spatial tokens (EVA-style) + stability controls:
    - Backbone frozen
    - One trainable token per block
    - Tokens init with timm.trunc_normal_(std=0.02)
    - Tokens L2-normalized + scaled each forward
    - Per-block gates (start at 0) + small prompt_scale to prevent overflow
    """

    def __init__(self, base_model, lock=True, token_std=0.02, prompt_scale=0.1):
        super().__init__()
        self.base = base_model
        self.is_locked = lock
        self.prompt_scale = prompt_scale  # global safety scale

        if lock:
            for p in self.base.parameters():
                p.requires_grad = False

        stage_configs = [
            (6, 3136, 64),  # 56x56
            (6, 784, 128),  # 28x28
            (18, 196, 256),  # 14x14
            (6, 49, 512),  # 7x7
        ]

        self.blocks_spatial_tokens = nn.ModuleList()
        self.token_gates = nn.ModuleList()

        for num_blocks, num_patches, embed_dim in stage_configs:
            stage_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, num_patches, embed_dim))
                    for _ in range(num_blocks)
                ]
            )
            for t in stage_tokens:
                trunc_normal_(t, std=token_std)

            stage_gates = nn.ParameterList(
                [nn.Parameter(torch.tensor(0.0)) for _ in range(num_blocks)]
            )

            self.blocks_spatial_tokens.append(stage_tokens)
            self.token_gates.append(stage_gates)

        total = sum(len(s) for s in self.blocks_spatial_tokens)
        print(f"Created {total} tokens + gates (trunc_normal init)")

        # Wrap each block forward to inject tokens before block runs
        for s_idx, stage in enumerate(self.base.stages):
            blocks = getattr(stage, "blocks", None)
            if blocks is None:
                continue

            for b_idx, blk in enumerate(blocks):
                if getattr(blk, "_has_token_wrap", False):
                    continue

                orig_fwd = blk.forward

                def fwd_with_token(
                    x, *args, orig_fwd=orig_fwd, s=s_idx, b=b_idx, **kwargs
                ):
                    x = self.add_spatial_token(x, s, b)
                    return orig_fwd(x, *args, **kwargs)

                blk.forward = fwd_with_token
                blk._has_token_wrap = True

    def add_spatial_token(self, x, stage_idx, block_idx):
        B, C, H, W = x.shape
        token = self.blocks_spatial_tokens[stage_idx][block_idx]  # (1, N, C)
        gate = self.token_gates[stage_idx][block_idx]  # scalar

        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        if token.shape[1] != x_flat.shape[1]:
            token_rs = token.transpose(1, 2)  # (1, C, N)
            token_rs = F.interpolate(token_rs, size=x_flat.shape[1], mode="linear")
            token = token_rs.transpose(1, 2)

        token_norm = F.normalize(token, p=2, dim=-1, eps=1e-8)
        token_scaled = token_norm * math.sqrt(token.shape[-1])

        alpha = torch.tanh(gate) * self.prompt_scale
        x_flat = x_flat + alpha * token_scaled

        return x_flat.transpose(1, 2).reshape(B, C, H, W)

    def forward_features(self, x, return_intermediates=False, stage_indices=None):
        """
        Default (return_intermediates=False): same as before -> timm forward_features.

        If return_intermediates=True:
            returns (final_feat_map, feats_dict) where feats_dict has
            'stage0'..'stage3' (after each stage) and 'final'.
        """
        if not return_intermediates:
            return self.base.forward_features(x)

        feats = {}

        # --- Stem / patch embedding ---
        if hasattr(self.base, "stem"):
            x = self.base.stem(x)
        elif hasattr(self.base, "patch_embed"):
            x = self.base.patch_embed(x)
        elif hasattr(self.base, "conv_stem"):
            x = self.base.conv_stem(x)

        # --- Stages (tokens are injected inside blocks) ---
        for s_idx, stage in enumerate(self.base.stages):
            x = stage(x)
            if stage_indices is None or s_idx in stage_indices:
                feats[f"stage{s_idx}"] = x

        # --- Apply final_conv (512 -> 1024 channels) ---
        if hasattr(self.base, "final_conv"):
            x = self.base.final_conv(x)

        # --- Final norm if present ---
        if hasattr(self.base, "norm"):
            x = self.base.norm(x)

        feats["final"] = x
        return x, feats

    def forward(self, x, return_intermediates=False, stage_indices=None):
        if return_intermediates:
            final_map, feats = self.forward_features(
                x, return_intermediates=True, stage_indices=stage_indices
            )
            return final_map, feats
        return self.base(x)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.is_locked:
            self.base.eval()
        return self
