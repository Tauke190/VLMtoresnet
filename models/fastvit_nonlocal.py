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

try:
    from .fastvit import FastViT, RepCPE, default_cfgs, RepMixerBlock, AttentionBlock
    from .modules.proposed_modules import Mlp, ConvAdapter, RepMixerBlock_Adapter, AttentionBlock_Adapter, ConvLoRA
except ImportError:
    # Fallback for debugging if modules aren't present
    print("Warning: Local modules not found. Ensure .fastvit and .modules are in the path.")
    FastViT = nn.Module
    RepCPE = RepMixerBlock = AttentionBlock = object
    default_cfgs = {"fastvit_m": {}}

import logging

_logger = logging.getLogger("train")
IMPORT_NONE = None

###### baseline with projectors
class FastViT_Projector(FastViT):
    def __init__(self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(**kwargs)

        if freeze_backbone:
            _logger.info("Freezing Backbone")
            for p in self.parameters():
                p.requires_grad = False
       
        self.projector = Mlp(in_features=self.head.in_features, out_features=clip_dim)
        self.apply(self.cls_init_weights)
   
    def load_state_dict(self, state_dict, strict=True):
        # Inject projector weights if missing (allows loading backbone-only checkpoints)
        prefix = "projector."
        if f"{prefix}fc1.weight" not in state_dict:
            state_dict[f"{prefix}fc1.weight"] = self.projector.fc1.weight
        if f"{prefix}fc1.bias" not in state_dict:
            state_dict[f"{prefix}fc1.bias"] = self.projector.fc1.bias
        if f"{prefix}fc2.weight" not in state_dict:
            state_dict[f"{prefix}fc2.weight"] = self.projector.fc2.weight
        if f"{prefix}fc2.bias" not in state_dict:
            state_dict[f"{prefix}fc2.bias"] = self.projector.fc2.bias

        super().load_state_dict(state_dict, strict)

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
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
        INITIAL_TOKENS = 32
        for idx in range(len(self.embed_dims)):
            p = nn.Parameter(torch.zeros(1, self.embed_dims[idx], INITIAL_TOKENS, INITIAL_TOKENS))
            # Xavier/Glorot uniform initialization adapted for tokens
            val = math.sqrt(6. / float(3 * reduce(mul, (INITIAL_TOKENS, INITIAL_TOKENS), 1) + self.embed_dims[idx]))
            nn.init.uniform_(p.data, -val, val)
            prompts.append(p)
            INITIAL_TOKENS = int(INITIAL_TOKENS // 2)
       
        self.deep_prompts = nn.ParameterList(prompts)

        self.mode = "bicubic"
        # Re-init projector weights
        self.projector.apply(self.cls_init_weights)
       
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")

    def load_state_dict(self, state_dict, strict=True):
        # Inject prompts if missing
        for idx, param in enumerate(self.deep_prompts):
            if f"deep_prompts.{idx}" not in state_dict:
                state_dict[f"deep_prompts.{idx}"] = param

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
                # Add prompt before the stage
                x = self._add_prompt(x, self.deep_prompts[stage_idx])
                stage_idx += 1
            
            x = block(x)
            
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
        
        for i, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                layer_index += 1
                for j, sub_block in enumerate(block):
                    block_index = layer_index
                    block_idx = j
                    drop_path_rate = kwargs.get('drop_path_rate', 0.0) # Default to 0.0 if not present
                    
                    # Calculate drop path rate safely
                    total_layers = sum(layers) if layers else 1
                    current_layer_sum = sum(layers[:block_index]) if layers else 0
                    block_dpr = (drop_path_rate * (block_idx + current_layer_sum) / (total_layers - 1))
                   
                    if isinstance(self.network[i][j], RepMixerBlock):
                        self.network[i][j] = RepMixerBlock_Adapter(
                            reduction_factor=adapter_reduction,
                            dim=embed_dims[layer_index], 
                            kernel_size=kwargs.get('repmixer_kernel_size', 3),
                            mlp_ratio=kwargs['mlp_ratios'][layer_index], 
                            act_layer=kwargs.get('act_layer', nn.GELU),
                            drop=kwargs.get('drop_rate', 0.0), 
                            use_layer_scale=kwargs.get('use_layer_scale', True), 
                            layer_scale_init_value=kwargs.get('layer_scale_init_value', 1e-5),
                            inference_mode=kwargs.get('inference_mode', False), 
                            drop_path=block_dpr
                        )
                    elif isinstance(self.network[i][j], AttentionBlock):
                        self.network[i][j] = AttentionBlock_Adapter(
                            reduction_factor=adapter_reduction,
                            dim=embed_dims[layer_index], 
                            mlp_ratio=kwargs['mlp_ratios'][layer_index],
                            act_layer=kwargs.get('act_layer', nn.GELU), 
                            drop=kwargs.get('drop_rate', 0.0),
                            drop_path=block_dpr, 
                            use_layer_scale=kwargs.get('use_layer_scale', True),
                            layer_scale_init_value=kwargs.get('layer_scale_init_value', 1e-5)
                        )
                    # Note: If it's neither, we just leave it alone (e.g. Identity or other layers)
                     
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Adapter reduction factor: {adapter_reduction}")

    def load_state_dict(self, state_dict, strict=True):
        # Initialize adapter weights if not in checkpoint (allows backward compatibility)
        for i, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                for j, sub_block in enumerate(block):
                    prefix = f"network.{i}.{j}"
                    
                    if isinstance(sub_block, RepMixerBlock_Adapter):
                        if f"{prefix}.adapter1.0.weight" not in state_dict:
                            state_dict[f"{prefix}.adapter1.0.bias"] = sub_block.adapter1[0].bias
                            state_dict[f"{prefix}.adapter1.0.weight"] = sub_block.adapter1[0].weight
                            state_dict[f"{prefix}.adapter1.2.bias"] = sub_block.adapter1[2].bias
                            state_dict[f"{prefix}.adapter1.2.weight"] = sub_block.adapter1[2].weight
                   
                    elif isinstance(sub_block, AttentionBlock_Adapter):
                        if f"{prefix}.adapter1.0.weight" not in state_dict:
                            state_dict[f"{prefix}.adapter1.0.bias"] = sub_block.adapter1[0].bias
                            state_dict[f"{prefix}.adapter1.0.weight"] = sub_block.adapter1[0].weight
                            state_dict[f"{prefix}.adapter2.0.bias"] = sub_block.adapter2[0].bias
                            state_dict[f"{prefix}.adapter2.0.weight"] = sub_block.adapter2[0].weight

                            state_dict[f"{prefix}.adapter1.2.bias"] = sub_block.adapter1[2].bias
                            state_dict[f"{prefix}.adapter1.2.weight"] = sub_block.adapter1[2].weight
                            state_dict[f"{prefix}.adapter2.2.bias"] = sub_block.adapter2[2].bias
                            state_dict[f"{prefix}.adapter2.2.weight"] = sub_block.adapter2[2].weight

        super().load_state_dict(state_dict, strict)

   
###### LoRA
class FastViT_lora(FastViT_Projector):
    def __init__(self, freeze_backbone=True, clip_dim=768, lora_rank=8, lora_alpha=16, **kwargs):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        # Create LoRA layers for each stage
        self.loras = nn.ModuleList()

        # Iterate over network to create corresponding LoRA layers
        current_dim_idx = 0
        
        for i, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                stage_loras = nn.ModuleList()
                for sub_block in block:
                    # detect dim safely
                    if hasattr(sub_block, "dim"):
                        dim = sub_block.dim
                    elif hasattr(sub_block, "in_channels"):
                         dim = sub_block.in_channels
                    else:
                        # Fallback to config embed_dims if attribute missing
                        dim = self.embed_dims[current_dim_idx]
                        
                    stage_loras.append(
                        ConvLoRA(
                            dim,
                            dim,
                            kernel_size=1,
                            rank=lora_rank,
                            alpha=lora_alpha
                        )
                    )
                self.loras.append(stage_loras)
                current_dim_idx += 1
       
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"LoRA rank: {lora_rank}, LoRA alpha: {lora_alpha}")

    def load_state_dict(self, state_dict, strict=True):
        # Initialize LoRA weights if not in checkpoint (allows backward compatibility)
        # self.loras is a ModuleList of ModuleLists: loras[stage_idx][block_idx]
        
        for stage_idx, stage_loras in enumerate(self.loras):
            for block_idx, lora_module in enumerate(stage_loras):
                # Construct key matching the ModuleList structure
                lora_prefix = f"loras.{stage_idx}.{block_idx}"
                
                # Check and inject if missing
                if f"{lora_prefix}.lora_A.weight" not in state_dict:
                    state_dict[f"{lora_prefix}.lora_A.weight"] = lora_module.lora_A.weight
                if f"{lora_prefix}.lora_B.weight" not in state_dict:
                    state_dict[f"{lora_prefix}.lora_B.weight"] = lora_module.lora_B.weight

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                # Apply LoRA after each sub-block in the sequence
                for block_idx, sub_block in enumerate(block):
                    x = sub_block(x)
                    # Apply corresponding LoRA
                    if stage_idx < len(self.loras) and block_idx < len(self.loras[stage_idx]):
                        x = self.loras[stage_idx][block_idx](x)
                stage_idx += 1
            else:
                x = block(x)

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

#### LoRA
@register_model
def fastvit_sa36_lora(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with LoRA"""
    model = FastViT_lora(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model
