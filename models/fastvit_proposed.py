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
from .modules.proposed_modules import Mlp, ConvAdapter, RepMixerBlock_Adapter, AttentionBlock_Adapter, ConvLoRA
from .modules.nonlocal_block import NonLocalBlock2d

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
                      
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Adapter reduction factor: {adapter_reduction}")

    def load_state_dict(self, state_dict, strict):
        # Initialize adapter weights if not in checkpoint (allows backward compatibility)
        layer_index = -1 
        for i,block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                for j,sub_block in enumerate(block):
                    # if one dapater is not present, all other wont be present either 
                    if f"network.{i}.{j}.adapter1.0.bias" not in state_dict:
                        state_dict[ f"network.{i}.{j}.adapter1.0.bias" ] = sub_block.adapter1[0].bias
                        state_dict[ f"network.{i}.{j}.adapter1.0.weight" ] = sub_block.adapter1[0].weight
                        state_dict[ f"network.{i}.{j}.adapter1.2.bias" ] = sub_block.adapter1[2].bias
                        state_dict[ f"network.{i}.{j}.adapter1.2.weight" ] = sub_block.adapter1[2].weight

                        if type(sub_block) == AttentionBlock_Adapter:
                            state_dict[ f"network.{i}.{j}.adapter2.0.bias" ] = sub_block.adapter2[0].bias
                            state_dict[ f"network.{i}.{j}.adapter2.0.weight" ] = sub_block.adapter2[0].weight
                            state_dict[ f"network.{i}.{j}.adapter2.2.bias" ] = sub_block.adapter2[2].bias
                            state_dict[ f"network.{i}.{j}.adapter2.2.weight" ] = sub_block.adapter2[2].weight

        
        super().load_state_dict(state_dict, strict)

    
###### LoRA 
class FastViT_lora(FastViT_Projector):
    def __init__(self, freeze_backbone=True, clip_dim=768, lora_rank=8, lora_alpha=16, **kwargs):
        super().__init__(**kwargs)

        # Create LoRA layers for each stage
        # Each stage outputs features with embed_dims[i] channels
        self.lora1 = ConvLoRA(self.embed_dims[0], self.embed_dims[0], kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.lora2 = ConvLoRA(self.embed_dims[1], self.embed_dims[1], kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.lora3 = ConvLoRA(self.embed_dims[2], self.embed_dims[2], kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.lora4 = ConvLoRA(self.embed_dims[3], self.embed_dims[3], kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.apply(self.cls_init_weights)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"LoRA rank: {lora_rank}, LoRA alpha: {lora_alpha}")

    def load_state_dict(self, state_dict, strict):
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

    def forward_tokens(self, x: torch.Tensor):
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

   
###### LoRA (priyank)
class FastViT_lora_PP(FastViT_Projector):
    def __init__(self, layers=-1, embed_dims=None, freeze_backbone=True, lora_rank=1, repmixer_kernel_size=3, **kwargs):
        super().__init__(layers=layers, embed_dims=embed_dims, repmixer_kernel_size=repmixer_kernel_size, **kwargs)

        from .modules.proposed_modules import StreightConv_LoRA, MergedLinear_LoRA

        self.layers = layers
        layer_index = -1 
        for i,block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                layer_index += 1
                for j,sub_block in enumerate(block):
                    dim = embed_dims[layer_index]
                    mlp_ratio = kwargs['mlp_ratios'][layer_index]
                    mlp_hidden_dim = int(dim * mlp_ratio)
                    # print(sub_block.convffn.fc2, mlp_hidden_dim, dim)
                    sub_block.convffn.fc2 = StreightConv_LoRA(in_features=mlp_hidden_dim, out_features=dim, r=lora_rank, bias=True, kernel_size=1)
                    
                    if type(self.network[i][j]) == RepMixerBlock:
                        assert len(sub_block.token_mixer.mixer.rbr_conv) == 1
                        # print(sub_block.token_mixer.mixer.rbr_conv[0][0], dim, dim)
                        sub_block.token_mixer.mixer.rbr_conv[0][0] = StreightConv_LoRA(in_features=dim, out_features=dim, r=lora_rank, kernel_size= repmixer_kernel_size, padding=repmixer_kernel_size // 2, groups=dim, bias=False)
                    elif type(self.network[i][j]) == AttentionBlock:
                        sub_block.token_mixer.qkv = MergedLinear_LoRA(dim, dim * 3, r=lora_rank, enable_lora=[True, True, True], bias=False)
        
        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"LoRA rank: {lora_rank}")

    def load_state_dict(self, state_dict, strict):
        # Initialize LoRA weights if not in checkpoint (allows backward compatibility)
        layer_index = -1
        for i,block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                for j,sub_block in enumerate(block):
                    if f"network.{i}.{j}.convffn.fc2.lora_A" not in state_dict:
                        # self.network[i][j].convffn.fc2.lora_A
                        state_dict[f"network.{i}.{j}.convffn.fc2.lora_A.weight"] = sub_block.convffn.fc2.lora_A.weight
                        state_dict[f"network.{i}.{j}.convffn.fc2.lora_B.weight"] = sub_block.convffn.fc2.lora_B.weight
                        if type(self.network[i][j]) == RepMixerBlock:
                            state_dict[f"network.{i}.{j}.token_mixer.mixer.rbr_conv.0.conv.lora_A.weight"] = sub_block.token_mixer.mixer.rbr_conv[0][0].lora_A.weight
                            state_dict[f"network.{i}.{j}.token_mixer.mixer.rbr_conv.0.conv.lora_B.weight"] = sub_block.token_mixer.mixer.rbr_conv[0][0].lora_B.weight
                                
                        elif type(self.network[i][j]) == AttentionBlock:
                            state_dict[f"network.{i}.{j}.token_mixer.qkv.lora_A"] = sub_block.token_mixer.qkv.lora_A
                            state_dict[f"network.{i}.{j}.token_mixer.qkv.lora_B"] = sub_block.token_mixer.qkv.lora_B

        super().load_state_dict(state_dict, strict)



    

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

#### LoRA (avinash)
@register_model
def fastvit_sa36_lora(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with LoRA"""
    model = FastViT_lora(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

#### LoRA (priyank)
@register_model
def fastvit_sa36_lora_pp(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with LoRA"""
    model = FastViT_lora_PP(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model


###### Non-Local Networks
class FastViT_nonlocal(FastViT_Projector):
    """FastViT with Non-Local Blocks inserted after every stage.
    
    Implements the non-local operation from:
        "Non-local Neural Networks" - Wang et al., CVPR 2018
        https://arxiv.org/abs/1711.07971
    
    A NonLocalBlock2d is applied after each of the 4 main stages
    (nn.Sequential blocks in self.network). This captures long-range
    dependencies at every spatial resolution of the hierarchy.
    
    The non-local blocks use:
    - Embedded Gaussian (softmax) attention with dim^{-0.5} scaling
    - Bottleneck: inter_channels = in_channels // 2
    - MaxPool spatial subsampling on phi and g (stride 2)
    - BatchNorm with gamma=0 init (identity at initialization)
    - Residual connection: z = x + NL(x)
    """
    def __init__(
        self,
        freeze_backbone=True,
        clip_dim=768,
        nl_inter_channels=None,
        nl_use_maxpool=True,
        nl_use_softmax=True,
        nl_use_scale=True,
        nl_use_bn=True,
        nl_bn_init_gamma=0.0,
        nl_conv_init_std=0.01,
        nl_max_pool_stride=2,
        **kwargs,
    ):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        # Create one NonLocalBlock2d per stage, matching each stage's channel dim
        self.nonlocal_blocks = nn.ModuleList()
        for dim in self.embed_dims:
            self.nonlocal_blocks.append(
                NonLocalBlock2d(
                    in_channels=dim,
                    inter_channels=nl_inter_channels if nl_inter_channels is not None else dim // 2,
                    use_maxpool=nl_use_maxpool,
                    use_softmax=nl_use_softmax,
                    use_scale=nl_use_scale,
                    use_bn=nl_use_bn,
                    bn_init_gamma=nl_bn_init_gamma,
                    conv_init_std=nl_conv_init_std,
                    max_pool_stride=nl_max_pool_stride,
                )
            )

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Non-Local blocks inserted after each of {len(self.embed_dims)} stages")
        _logger.info(f"Non-Local inter_channels: {[blk.inter_channels for blk in self.nonlocal_blocks]}")

    def load_state_dict(self, state_dict, strict):
        """Handle loading checkpoints that don't contain non-local block weights.
        
        If non-local block parameters are missing from the checkpoint
        (e.g., loading a vanilla FastViT pretrained model), fill them from
        the current model's initialized values for backward compatibility.
        """
        for i, nl_block in enumerate(self.nonlocal_blocks):
            for name, param in nl_block.state_dict().items():
                key = f"nonlocal_blocks.{i}.{name}"
                if key not in state_dict:
                    state_dict[key] = param

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        """Run self.network, applying a Non-Local block after each stage.
        
        Each nn.Sequential in self.network is a stage. After each stage,
        the corresponding NonLocalBlock2d is applied to capture long-range
        spatial dependencies at that resolution.
        """
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)

            # Each main stage is an nn.Sequential - apply non-local after it
            if isinstance(block, nn.Sequential):
                x = self.nonlocal_blocks[stage_idx](x)
                stage_idx += 1

            # Collect intermediate outputs if needed for dense tasks
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))

        if self.fork_feat:
            return outs
        return x


#### Non-Local
@register_model
def fastvit_sa36_nonlocal(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with Non-Local blocks."""
    model = FastViT_nonlocal(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model







