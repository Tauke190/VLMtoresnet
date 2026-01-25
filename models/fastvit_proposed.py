import os
import copy
import math 
from functools import partial, reduce
from operator import mul
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from .fastvit import FastViT, RepCPE, default_cfgs, RepMixerBlock, AttentionBlock
from .modules.proposed_modules import Mlp, ConvAdapter, RepMixerBlock_Adapter, AttentionBlock_Adapter

import logging

_logger = logging.getLogger("train")
IMPORT_NONE = None 

###### baseline with projectors 
class FastViT_Projector(FastViT):
    def __init__(self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(**kwargs)

        if freeze_backbone:
            _logger.info(" Frezing Backbone")
            for p in self.parameters():
                p.requires_grad = False
       
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

        super().load_state_dict(state_dict, strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        # for image classification
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        projected_embed = self.projector(x)

        return projected_embed, cls_out, x

        
        
###### Lr-tokens 
class FastViT_lrtokens(FastViT_Projector):
    def __init__(self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(**kwargs)
        
        prompts = []
        INTIAL_TOKENS = 32 
        for idx in range( len(self.embed_dims) ):
            p = nn.Parameter(torch.zeros(1, self.embed_dims[idx], INTIAL_TOKENS, INTIAL_TOKENS))
            val = math.sqrt(6. / float(3 * reduce(mul, (INTIAL_TOKENS, INTIAL_TOKENS) , 1) + self.embed_dims[idx]))  # noqa
            nn.init.uniform_(p.data, -val, val)
            prompts.append(p)
            INTIAL_TOKENS = int(INTIAL_TOKENS // 2)
        
        self.deep_pompts = nn.ParameterList(prompts)

        self.mode = "bicubic"
        self.apply(self.cls_init_weights)
        
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        print("Number of nn.Sequential blocks in self.network:", num_sequential)

    def load_state_dict(self, state_dict, strict):
        # for e in self.state_dict().keys():e
        for idx, param in enumerate(self.deep_pompts):
            if f"deep_pompts.{idx}" not in state_dict:
                state_dict[f"deep_pompts.{idx}"] = param

        super().load_state_dict(state_dict, strict)

    def _add_prompt(self, x, p):
        # x: [B,C,H,W] -> feature map , p: [1,C,h0,w0] -> learnable tokens
        H, W = x.shape[-2:]
        p_up = nn.functional.interpolate(p, size=(H, W), mode=self.mode,
                             align_corners=False if self.mode in ("bilinear", "bicubic") else None)
        return x + p_up  # broadcast over batch
        
    def forward_tokens(self, x: torch.Tensor):
        """
        Run self.network, adding p1..p4 just before each of the 4 main stages.
        """
        outs = []
        stage_idx = 0 
        for idx, block in enumerate(self.network):

            if isinstance(block, nn.Sequential):
                # print(self.deep_pompts[stage_idx].shape, x.shape)
                x = self._add_prompt(x, self.deep_pompts[stage_idx])
                stage_idx += 1 
            x = block(x)
            # print(x.shape)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    

###### Adapters
class FastViT_adapter(FastViT_Projector):
    def __init__(self, layers=-1, embed_dims=None, freeze_backbone=True, adapter_reduction=4, **kwargs):
        super().__init__(layers=layers, embed_dims=embed_dims, **kwargs)

        self.layers = layers
        layer_index = -1 
        for i,block in enumerate(self.network):
            # isinstance(block, nn.Sequential)
            if isinstance(block, nn.Sequential):
                layer_index += 1
                for j,sub_block in enumerate(block):
                    block_index = layer_index
                    block_idx = j 
                    drop_path_rate=kwargs.get('drop_path_rate', True)
                    block_dpr = ( drop_path_rate * (block_idx + sum(layers[:block_index])) / (sum(layers) - 1) )
                    
                    if type(self.network[i][j]) == RepMixerBlock:
                        self.network[i][j] = RepMixerBlock_Adapter( reduction_factor=adapter_reduction, 
                            dim=embed_dims[layer_index], kernel_size=kwargs.get('repmixer_kernel_size', 3),
                            mlp_ratio=kwargs['mlp_ratios'][layer_index], act_layer=kwargs.get('act_layer', nn.GELU),
                            drop=kwargs.get('drop_rate', True), use_layer_scale=kwargs.get('use_layer_scale', True), layer_scale_init_value=kwargs.get('layer_scale_init_value', 1e-5),
                            inference_mode=kwargs.get('inference_mode', False), drop_path=block_dpr)
                    elif type(self.network[i][j]) == AttentionBlock:
                        self.network[i][j] = AttentionBlock_Adapter( reduction_factor=adapter_reduction, 
                            dim=embed_dims[layer_index],  mlp_ratio=kwargs['mlp_ratios'][layer_index], 
                            act_layer=kwargs.get('act_layer', nn.GELU), drop=kwargs.get('drop_rate', True), 
                            drop_path=block_dpr, use_layer_scale=kwargs.get('use_layer_scale', True), 
                            layer_scale_init_value=kwargs.get('layer_scale_init_value', 1e-5))
                    else:
                        assert False, "module not recognized"
                      
        # # Create adapter modules for each stage
        # self.adapter1 = ConvAdapter(self.embed_dims[0], reduction_factor=adapter_reduction)
        # self.adapter2 = ConvAdapter(self.embed_dims[1], reduction_factor=adapter_reduction)
        # self.adapter3 = ConvAdapter(self.embed_dims[2], reduction_factor=adapter_reduction)
        # self.adapter4 = ConvAdapter(self.embed_dims[3], reduction_factor=adapter_reduction)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Adapter reduction factor: {adapter_reduction}")

    def load_state_dict(self, state_dict, strict):
        # Initialize adapter weights if not in checkpoint (allows backward compatibility)
        layer_index = -1 
        for i,block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                for j,sub_block in enumerate(block):
                    if type(sub_block) == RepMixerBlock_Adapter:
                        state_dict[ f"network.{i}.{j}.adapter1.0.bias" ] = sub_block.adapter1[0].bias
                        state_dict[ f"network.{i}.{j}.adapter1.0.weight" ] = sub_block.adapter1[0].weight

                        state_dict[ f"network.{i}.{j}.adapter1.2.bias" ] = sub_block.adapter1[2].bias
                        state_dict[ f"network.{i}.{j}.adapter1.2.weight" ] = sub_block.adapter1[2].weight
                    
                    elif type(sub_block) == AttentionBlock_Adapter:
                        state_dict[ f"network.{i}.{j}.adapter1.0.bias" ] = sub_block.adapter1[0].bias
                        state_dict[ f"network.{i}.{j}.adapter1.0.weight" ] = sub_block.adapter1[0].weight
                        state_dict[ f"network.{i}.{j}.adapter2.0.bias" ] = sub_block.adapter2[0].bias
                        state_dict[ f"network.{i}.{j}.adapter2.0.weight" ] = sub_block.adapter2[0].weight

                        state_dict[ f"network.{i}.{j}.adapter1.2.bias" ] = sub_block.adapter1[2].bias
                        state_dict[ f"network.{i}.{j}.adapter1.2.weight" ] = sub_block.adapter1[2].weight
                        state_dict[ f"network.{i}.{j}.adapter2.2.bias" ] = sub_block.adapter2[2].bias
                        state_dict[ f"network.{i}.{j}.adapter2.2.weight" ] = sub_block.adapter2[2].weight

        # import pdb
        # pdb.set_trace()                
        # for stage_num in range(1, 5):
        #     adapter_name = f"adapter{stage_num}"
        #     adapter_module = getattr(self, adapter_name)
        #     if f"{adapter_name}.adapter.0.weight" not in state_dict:
        #         state_dict[f"{adapter_name}.adapter.0.weight"] = adapter_module.adapter[0].weight
        #     if f"{adapter_name}.adapter.0.bias" not in state_dict:
        #         state_dict[f"{adapter_name}.adapter.0.bias"] = adapter_module.adapter[0].bias
        #     if f"{adapter_name}.adapter.2.weight" not in state_dict:
        #         state_dict[f"{adapter_name}.adapter.2.weight"] = adapter_module.adapter[2].weight
        #     if f"{adapter_name}.adapter.2.bias" not in state_dict:
        #         state_dict[f"{adapter_name}.adapter.2.bias"] = adapter_module.adapter[2].bias

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        """
        Run self.network, applying adapters after each of the 4 main stages.
        """
        import pdb
        pdb.set_trace()
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

    
    

fastvit_sa36_config = dict(
    layers = [6, 6, 18, 6],
    embed_dims = [64, 128, 256, 512],
    mlp_ratios = [4, 4, 4, 4],
    downsamples = [True, True, True, True],
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention"),
    layer_scale_init_value=1e-6,
)

#### Projector
@register_model
def fastvit_sa36_projector(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with adapters"""
    model = FastViT_Projector(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

#### lr tokens
@register_model
def fastvit_sa36_lrtokens(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    model = FastViT_lrtokens(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

#### Adapters
@register_model
def fastvit_sa36_adapter(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with adapters"""
    model = FastViT_adapter(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model




