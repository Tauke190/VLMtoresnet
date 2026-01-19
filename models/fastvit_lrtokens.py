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

class FastViT_lrtokens(FastViT):
    def __init__(self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(**kwargs)
        
        if freeze_backbone:
            _logger.info(" Frezing Backbone")
            for p in self.parameters():
                p.requires_grad = False
       
        self.p1 = nn.Parameter(torch.zeros(1, self.embed_dims[0], 32, 32))
        self.p2 = nn.Parameter(torch.zeros(1, self.embed_dims[1], 16, 16))
        self.p3 = nn.Parameter(torch.zeros(1, self.embed_dims[2], 8, 8))
        self.p4 = nn.Parameter(torch.zeros(1, self.embed_dims[3], 4, 4))
        self.mode = "bicubic"
        
        self.projector = Mlp(in_features=self.head.in_features, out_features=clip_dim)
        self.apply(self.cls_init_weights)
        
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        print("Number of nn.Sequential blocks in self.network:", num_sequential)

    def load_state_dict(self, state_dict, strict):
        if "projector.fc1.weight" not in state_dict:
            state_dict["projector.fc1.weight"] = self.projector.fc1.weight
        if "projector.fc1.bias" not in state_dict:
            state_dict["projector.fc1.bias"] = self.projector.fc1.bias
        if "projector.fc2.weight" not in state_dict:
            state_dict["projector.fc2.weight"] = self.projector.fc2.weight
        if "projector.fc2.bias" not in state_dict:
            state_dict["projector.fc2.bias"] = self.projector.fc2.bias

        if "p1" not in state_dict:
            state_dict["p1"] = self.p1
        if "p2" not in state_dict:
            state_dict["p2"] = self.p2
        if "p3" not in state_dict:
            state_dict["p3"] = self.p3
        if "p4" not in state_dict:
            state_dict["p4"] = self.p4

        super().load_state_dict(state_dict, strict)

    def _add_prompt(self, x, p):
        # x: [B,C,H,W] -> feature map , p: [1,C,h0,w0] -> learnable tokens
        H, W = x.shape[-2:]
        p_up = nn.functional.interpolate(p, size=(H, W), mode=self.mode,
                             align_corners=False if self.mode in ("bilinear", "bicubic") else None)
        return x + p_up  # broadcast over batch
        
    def _forward_tokens_with_prompts(self, x: torch.Tensor):
        """
        Run self.network, adding p1..p4 just before each of the 4 main stages.
        """
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            # Each main stage is an nn.Sequential
            if isinstance(block, nn.Sequential):
                # add prompt before this stage
                if stage_idx == 0:
                    x = self._add_prompt(x, self.p1)
                elif stage_idx == 1:
                    x = self._add_prompt(x, self.p2)
                elif stage_idx == 2:
                    x = self._add_prompt(x, self.p3)
                elif stage_idx == 3:
                    x = self._add_prompt(x, self.p4)
                stage_idx += 1

            x = block(x)

            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))

        if self.fork_feat:
            return outs
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Convolutional stem
        x = self.forward_embeddings(x)

        # 2) Backbone with prompts injected before stages
        x = self._forward_tokens_with_prompts(x)

        if self.fork_feat:
            return x  # or outs if you use fork_feat for dense tasks

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
def fastvit_sa36_lrtokens(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    model = FastViT_projector(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model
