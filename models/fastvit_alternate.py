import math
import logging
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from .fastvit import FastViT, RepCPE, default_cfgs, RepMixerBlock, AttentionBlock
from .modules.proposed_modules import (
    Mlp,
    ConvAdapter,
    RepMixerBlock_Adapter,
    AttentionBlock_Adapter,
    ConvLoRA
)

from fastvit_proposed import FastViT_Projector

_logger = logging.getLogger("train")

###### LoRA 
class FastViT_lora(FastViT_Projector):
    def __init__(self, freeze_backbone=True, clip_dim=768, lora_rank=8, lora_alpha=16, **kwargs):
        super().__init__(
            freeze_backbone=freeze_backbone,
            clip_dim=clip_dim,
            **kwargs,
        )
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.lora_layers = nn.ModuleList()
        for dim in self.embed_dims:
            self.lora_layers.append(
                ConvLoRA(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=1,
                    rank=lora_rank,
                    alpha=lora_alpha,
                )
            )

        self.apply(self.cls_init_weights)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")

    def load_state_dict(self, state_dict, strict):
        for i, lora in enumerate(self.lora_layers):
            for name, param in lora.state_dict().items():
                key = f"lora_layers.{i}.{name}"
                if key not in state_dict:
                    state_dict[key] = param

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        outs = []
        stage_idx = 0

        for idx, block in enumerate(self.network):

            if isinstance(block, nn.Sequential):
                x = block(x)
                x = self.lora_layers[stage_idx](x)
                stage_idx += 1

            else:
                x = block(x)

            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)

        if self.fork_feat:
            return outs
        return x

fastvit_sa36_config = dict(
    layers=[6, 6, 18, 6],
    embed_dims=[64, 128, 256, 512],
    mlp_ratios=[4, 4, 4, 4],
    downsamples=[True, True, True, True],
    pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
    token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    layer_scale_init_value=1e-6,
)

#### LoRA
@register_model
def fastvit_sa36_lora(pretrained=False, **kwargs):
    model = FastViT_lora(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model