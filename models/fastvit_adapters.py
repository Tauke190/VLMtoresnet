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

class ConvAdapter(nn.Module):
    """
    Convolutional Adapter module for efficient parameter tuning.
    Uses a bottleneck architecture: down-projection -> activation -> up-projection.
    """
    def __init__(self, in_channels, reduction_factor=4):
        super().__init__()
        hidden_dim = max(in_channels // reduction_factor, 8)

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)
        )

        # Initialize to near-zero so adapter starts as identity
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        return x + self.adapter(x)  # Residual connection 

class FastViT_adapter(FastViT):
    def __init__(self, freeze_backbone=True, clip_dim=768, adapter_reduction=4, **kwargs):
        super().__init__(**kwargs)

        if freeze_backbone:
            _logger.info(" Freezing Backbone")
            for p in self.parameters():
                p.requires_grad = False

        # Create adapter modules for each stage
        self.adapter1 = ConvAdapter(self.embed_dims[0], reduction_factor=adapter_reduction)
        self.adapter2 = ConvAdapter(self.embed_dims[1], reduction_factor=adapter_reduction)
        self.adapter3 = ConvAdapter(self.embed_dims[2], reduction_factor=adapter_reduction)
        self.adapter4 = ConvAdapter(self.embed_dims[3], reduction_factor=adapter_reduction)

        self.projector = Mlp(in_features=self.head.in_features, out_features=clip_dim)
        self.apply(self.cls_init_weights)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Adapter reduction factor: {adapter_reduction}")

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

        # Initialize adapter weights if not in checkpoint (allows backward compatibility)
        for stage_num in range(1, 5):
            adapter_name = f"adapter{stage_num}"
            adapter_module = getattr(self, adapter_name)
            if f"{adapter_name}.adapter.0.weight" not in state_dict:
                state_dict[f"{adapter_name}.adapter.0.weight"] = adapter_module.adapter[0].weight
            if f"{adapter_name}.adapter.0.bias" not in state_dict:
                state_dict[f"{adapter_name}.adapter.0.bias"] = adapter_module.adapter[0].bias
            if f"{adapter_name}.adapter.2.weight" not in state_dict:
                state_dict[f"{adapter_name}.adapter.2.weight"] = adapter_module.adapter[2].weight
            if f"{adapter_name}.adapter.2.bias" not in state_dict:
                state_dict[f"{adapter_name}.adapter.2.bias"] = adapter_module.adapter[2].bias

        super().load_state_dict(state_dict, strict)

    def _forward_tokens_with_adapters(self, x: torch.Tensor):
        """
        Run self.network, applying adapters after each of the 4 main stages.
        """
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            # Process the block (embedding/downsample or main stage)
            x = block(x)

            # Each main stage is an nn.Sequential - apply adapter after it
            if isinstance(block, nn.Sequential):
                if stage_idx == 0:
                    x = self.adapter1(x)
                elif stage_idx == 1:
                    x = self.adapter2(x)
                elif stage_idx == 2:
                    x = self.adapter3(x)
                elif stage_idx == 3:
                    x = self.adapter4(x)
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

        # 2) Backbone with adapters applied after each stage
        x = self._forward_tokens_with_adapters(x)

        if self.fork_feat:
            return x  # Return intermediate features for dense tasks

        # 3) Classification head
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        projected_embed = self.projector(x)
        cls_out = self.head(x)
        return projected_embed, cls_out, x

    def get_stage_inputs(self, x: torch.Tensor):
        """From raw image -> feature maps before each stage (for debugging)."""
        x = self.forward_embeddings(x)
        _, stage_inputs = self._forward_backbone_with_stage_inputs(x)
        return stage_inputs

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
def fastvit_sa36_adapter(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with adapters"""
    model = FastViT_adapter(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model
