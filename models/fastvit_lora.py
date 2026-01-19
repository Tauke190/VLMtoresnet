import os
import copy
from functools import partial
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from .fastvit import FastViT, RepCPE, default_cfgs
from .modules.proposed_modules import Mlp

import logging
_logger = logging.getLogger("train")
IMPORT_NONE = None

class ConvLoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) for convolutional layers.
    Applies low-rank decomposition: output = Conv(x) + scale * (B * A)(x)
    where A projects down to rank r, and B projects back up.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, rank=8, alpha=16, stride=1, padding=0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scaling factor for LoRA

        # Low-rank matrices: A (down-projection) and B (up-projection)
        # A: [in_channels, rank, kernel_size, kernel_size]
        # B: [out_channels, rank, 1, 1]
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        # Initialize A with small random values (Kaiming), B with zeros
        # This ensures LoRA starts as identity (no effect initially)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # LoRA path: x -> A (down) -> B (up)
        lora_out = self.lora_B(self.lora_A(x))
        return x + self.scaling * lora_out  # Residual connection with scaling


class FastViT_lora(FastViT):
    def __init__(self, freeze_backbone=True, clip_dim=768, lora_rank=8, lora_alpha=16, **kwargs):
        super().__init__(**kwargs)

        if freeze_backbone:
            _logger.info(" Freezing Backbone")
            for p in self.parameters():
                p.requires_grad = False

        # Create LoRA layers for each stage
        # Each stage outputs features with embed_dims[i] channels
        self.lora1 = ConvLoRA(self.embed_dims[0], self.embed_dims[0],
                              kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.lora2 = ConvLoRA(self.embed_dims[1], self.embed_dims[1],
                              kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.lora3 = ConvLoRA(self.embed_dims[2], self.embed_dims[2],
                              kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.lora4 = ConvLoRA(self.embed_dims[3], self.embed_dims[3],
                              kernel_size=1, rank=lora_rank, alpha=lora_alpha)

        self.projector = Mlp(in_features=self.head.in_features, out_features=clip_dim)
        self.apply(self.cls_init_weights)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"LoRA rank: {lora_rank}, LoRA alpha: {lora_alpha}")

    def load_state_dict(self, state_dict, strict):
        # Initialize projector weights if not in checkpoint
        if "projector.fc1.weight" not in state_dict:
            state_dict["projector.fc1.weight"] = self.projector.fc1.weight
        if "projector.fc1.bias" not in state_dict:
            state_dict["projector.fc1.bias"] = self.projector.fc1.bias
        if "projector.fc2.weight" not in state_dict:
            state_dict["projector.fc2.weight"] = self.projector.fc2.weight
        if "projector.fc2.bias" not in state_dict:
            state_dict["projector.fc2.bias"] = self.projector.fc2.bias

        # Initialize LoRA weights if not in checkpoint (allows backward compatibility)
        for stage_num in range(1, 5):
            lora_name = f"lora{stage_num}"
            lora_module = getattr(self, lora_name)
            # LoRA has two matrices: A (down-projection) and B (up-projection)
            if f"{lora_name}.lora_A.weight" not in state_dict:
                state_dict[f"{lora_name}.lora_A.weight"] = lora_module.lora_A.weight
            if f"{lora_name}.lora_B.weight" not in state_dict:
                state_dict[f"{lora_name}.lora_B.weight"] = lora_module.lora_B.weight

        super().load_state_dict(state_dict, strict)

    def _forward_tokens_with_lora(self, x: torch.Tensor):
        """
        Run self.network, applying LoRA after each of the 4 main stages.
        """
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            # Process the block (embedding/downsample or main stage)
            x = block(x)

            # Each main stage is an nn.Sequential - apply LoRA after it
            if isinstance(block, nn.Sequential):
                if stage_idx == 0:
                    x = self.lora1(x)
                elif stage_idx == 1:
                    x = self.lora2(x)
                elif stage_idx == 2:
                    x = self.lora3(x)
                elif stage_idx == 3:
                    x = self.lora4(x)
                stage_idx += 1

            # Collect intermediate outputs if needed for dense tasks
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))

        if self.fork_feat:
            return outs
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Convolutional stem
        x = self.forward_embeddings(x)

        # 2) Backbone with LoRA applied after each stage
        x = self._forward_tokens_with_lora(x)

        if self.fork_feat:
            return x  # Return intermediate features for dense tasks

        # 3) Classification head
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        projected_embed = self.projector(x)
        cls_out = self.head(x)
        return projected_embed, cls_out, x


fastvit_sa36_config = dict(
    layers = [6, 6, 18, 6],
    embed_dims = [64, 128, 256, 512],
    mlp_ratios = [4, 4, 4, 4],
    downsamples = [True, True, True, True],
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention"),
    layer_scale_init_value=1e-6,
)


@register_model
def fastvit_sa36_lora(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with LoRA"""
    model = FastViT_lora(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model
