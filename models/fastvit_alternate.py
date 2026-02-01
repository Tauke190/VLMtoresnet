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
)

_logger = logging.getLogger("train")

###### baseline with projectors 
class FastViT_Projector(FastViT):
    def __init__(self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(**kwargs)

        if freeze_backbone:
            _logger.info("Freezing backbone")
            for p in self.parameters():
                p.requires_grad = False

        self.projector = Mlp(in_features=self.head.in_features, out_features=clip_dim)
        self.apply(self.cls_init_weights)

    def forward(self, x: torch.Tensor):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        cls_out = self.head(x)
        proj_out = self.projector(x)

        return proj_out, cls_out, x

###### Lr-tokens 
class FastViT_lrtokens(FastViT_Projector):
    def __init__(self, freeze_backbone=True, clip_dim=768, **kwargs):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        prompts = []
        INTIAL_TOKENS = 32
        for idx in range(len(self.embed_dims)):
            p = nn.Parameter(torch.zeros(1, self.embed_dims[idx], INTIAL_TOKENS, INTIAL_TOKENS))
            val = math.sqrt(6. / float(3 * reduce(mul, (INTIAL_TOKENS, INTIAL_TOKENS), 1) + self.embed_dims[idx]))
            nn.init.uniform_(p.data, -val, val)
            prompts.append(p)
            INTIAL_TOKENS //= 2

        self.deep_prompts = nn.ParameterList(prompts)
        self.mode = "bicubic"
        self.apply(self.cls_init_weights)

    def _add_prompt(self, x, p):
        H, W = x.shape[-2:]
        p_up = nn.functional.interpolate(
            p, size=(H, W), mode=self.mode,
            align_corners=False if self.mode in ("bilinear", "bicubic") else None
        )
        return x + p_up

    def forward_tokens(self, x: torch.Tensor):
        outs = []
        stage_idx = 0
        for idx, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                x = self._add_prompt(x, self.deep_prompts[stage_idx])
                stage_idx += 1
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))
        return outs if self.fork_feat else x

###### Adapters
class FastViT_adapter(FastViT_Projector):
    def __init__(self, layers=-1, embed_dims=None, freeze_backbone=True,
                 adapter_reduction=4, **kwargs):
        super().__init__(layers=layers, embed_dims=embed_dims,
                         freeze_backbone=freeze_backbone, **kwargs)

        self.adapter_layers = []

        layer_index = -1
        for i, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                layer_index += 1
                for j, sub in enumerate(block):
                    dpr = kwargs.get("drop_path_rate", 0.0)
                    block_dpr = dpr * (j + sum(layers[:layer_index])) / max(1, sum(layers) - 1)

                    if isinstance(sub, RepMixerBlock):
                        new_block = RepMixerBlock_Adapter(
                            dim=embed_dims[layer_index],
                            reduction_factor=adapter_reduction,
                            kernel_size=kwargs.get("repmixer_kernel_size", 3),
                            mlp_ratio=kwargs["mlp_ratios"][layer_index],
                            act_layer=kwargs.get("act_layer", nn.GELU),
                            drop=kwargs.get("drop_rate", 0.),
                            use_layer_scale=kwargs.get("use_layer_scale", True),
                            layer_scale_init_value=kwargs.get("layer_scale_init_value", 1e-5),
                            inference_mode=kwargs.get("inference_mode", False),
                            drop_path=block_dpr,
                        )
                        self.network[i][j] = new_block
                        self.adapter_layers.append(new_block)

                    elif isinstance(sub, AttentionBlock):
                        new_block = AttentionBlock_Adapter(
                             dim=embed_dims[layer_index],
                            reduction_factor=adapter_reduction,
                            mlp_ratio=kwargs["mlp_ratios"][layer_index],
                            act_layer=kwargs.get("act_layer", nn.GELU),
                            drop=kwargs.get("drop_rate", 0.),
                            drop_path=block_dpr,
                            use_layer_scale=kwargs.get("use_layer_scale", True),
                            layer_scale_init_value=kwargs.get("layer_scale_init_value", 1e-5),
                        )
                        self.network[i][j] = new_block
                        self.adapter_layers.append(new_block)

        _logger.info(f"Injected adapters into {len(self.adapter_layers)} blocks")

###### LoRA 
class FastViT_lora(FastViT_Projector):
    def __init__(self, freeze_backbone=True, clip_dim=768,
                 lora_reduction=4, **kwargs):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        self.lora_layers = []

        for i, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                for j, sub in enumerate(block):
                    if isinstance(sub, RepMixerBlock) and hasattr(sub, "convffn"):
                        if hasattr(sub.convffn, "fc1"):
                            hidden_channels = sub.convffn.fc1.out_channels
                            adapter = ConvAdapter(
                                in_channels=hidden_channels,
                                reduction_factor=lora_reduction
                            )
                            original_forward = sub.convffn.forward
                            def make_forward_with_adapter(orig_forward, adapt):
                                def forward_with_adapter(x):
                                    x = orig_forward.__self__.conv(x)
                                    x = orig_forward.__self__.fc1(x)
                                    x = orig_forward.__self__.act(x)
                                    x = orig_forward.__self__.drop(x)
                                    x = adapt(x)
                                    x = orig_forward.__self__.fc2(x)
                                    x = orig_forward.__self__.drop(x)
                                    return x
                                return forward_with_adapter
                            sub.convffn.forward = make_forward_with_adapter(original_forward, adapter)
                            sub.convffn.adapter = adapter  
                            
                            self.lora_layers.append(adapter)
                            
                    if isinstance(sub, AttentionBlock):
                        if hasattr(sub, "convffn") and hasattr(sub.convffn, "fc1"):
                            hidden_channels = sub.convffn.fc1.out_channels
                            adapter = ConvAdapter(
                                in_channels=hidden_channels,
                                reduction_factor=lora_reduction
                            )
                            original_forward = sub.convffn.forward
                            def make_forward_with_adapter(orig_forward, adapt):
                                def forward_with_adapter(x):
                                    x = orig_forward.__self__.conv(x)
                                    x = orig_forward.__self__.fc1(x)
                                    x = orig_forward.__self__.act(x)
                                    x = orig_forward.__self__.drop(x)
                                    x = adapt(x)
                                    x = orig_forward.__self__.fc2(x)
                                    x = orig_forward.__self__.drop(x)
                                    return x
                                return forward_with_adapter
                
                            sub.convffn.forward = make_forward_with_adapter(original_forward, adapter)
                            sub.convffn.adapter = adapter
                            
                            self.lora_layers.append(adapter)

        _logger.info(f"Injected LoRA adapters into {len(self.lora_layers)} layers")

fastvit_sa36_config = dict(
    layers=[6, 6, 18, 6],
    embed_dims=[64, 128, 256, 512],
    mlp_ratios=[4, 4, 4, 4],
    downsamples=[True, True, True, True],
    pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
    token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    layer_scale_init_value=1e-6,
)

#### Projector
@register_model
def fastvit_sa36_projector(pretrained=False, **kwargs):
    model = FastViT_Projector(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

#### LR Tokens
@register_model
def fastvit_sa36_lrtokens(pretrained=False, **kwargs):
    model = FastViT_lrtokens(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

#### Adapters
@register_model
def fastvit_sa36_adapter(pretrained=False, **kwargs):
    model = FastViT_adapter(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

#### LoRA
@register_model
def fastvit_sa36_lora(pretrained=False, **kwargs):
    model = FastViT_lora(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model