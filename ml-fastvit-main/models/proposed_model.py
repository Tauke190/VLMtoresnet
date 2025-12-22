import os
import copy
from functools import partial
from typing import List, Tuple, Optional, Union
from Functions.AddLRtokens import AddLRtokens
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from .fastvit import FastViT, RepCPE, default_cfgs
from .modules.proposed_modules import Mlp

import logging
_logger = logging.getLogger("train")
IMPORT_NONE = None 

class FastViT_projector(FastViT):
    def __init__( self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(**kwargs)
        

        if freeze_backbone:
            _logger.info(" Frezing Backbone")
            for p in self.parameters():
                p.requires_grad = False


        self.prompter = AddLRtokens(
            self.embed_dims[0], self.embed_dims[1], self.embed_dims[2], self.embed_dims[3],
            base_hw=(32,32), mode="bicubic"
        )
        self.projector = Mlp(in_features=self.head.in_features, out_features=clip_dim)
        self.apply(self.cls_init_weights)
        
    def load_state_dict(self, state_dict, strict):
        if "projector.fc1.weight" not in state_dict:
            state_dict["projector.fc1.weight"] = self.projector.fc1.weight
        if "projector.fc1.bias" not in state_dict:
            state_dict["projector.fc1.bias"] = self.projector.fc1.bias
        if "projector.fc2.weight" not in state_dict:
            state_dict["projector.fc2.weight"] = self.projector.fc2.weight
        if "projector.fc2.bias" not in state_dict:
            state_dict["projector.fc2.bias"] = self.projector.fc2.bias

        for i in range(1, 5):
            p = getattr(self.prompter, f"p{i}", None)
            if p is not None:
                if f"prompter.p{i}" not in state_dict:
                    state_dict[f"prompter.p{i}"] = p

        super().load_state_dict(state_dict, strict)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Convolutional stem
        x = self.forward_embeddings(x)

        # 2) backbone , get inputs before each stage
        x, stage_inputs = self.forward_tokens(x)

        f1, f2, f3, f4 = stage_inputs
        # 3) apply prompts
        f1, f2, f3, f4 = self.prompter(f1, f2, f3, f4)


        if self.fork_feat:
            return x # features after the stage

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
def fastvit_sa36_projector(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    model = FastViT_projector(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

