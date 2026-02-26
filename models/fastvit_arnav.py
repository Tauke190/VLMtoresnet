import torch
import torch.nn as nn

from timm.models.registry import register_model

from .fastvit import FastViT, default_cfgs
from .fastvit_proposed import FastViT_Projector, fastvit_sa36_config
from .modules.nonlocal_block import NonLocalBlock2d, NonLocalBlockLinear, MultiHeadNonLocalBlock2d

import logging

_logger = logging.getLogger("train")


###### Non-Local Networks
class FastViT_nonlocal(FastViT_Projector):
    def __init__(
        self,
        freeze_backbone=True,
        clip_dim=768,
        nl_inter_channels=None,
        nl_bn_layer=True,
        **kwargs,
    ):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        # Create NonLocal blocks:
        #   Stages 0-2 (first 3): NonLocalBlock2d  (Conv2d projections)
        #   Stage  3   (last):     NonLocalBlockLinear (Linear projections)
        self.nonlocal_blocks = nn.ModuleList()
        for stage_idx, dim in enumerate(self.embed_dims):
            inter_ch = nl_inter_channels if nl_inter_channels is not None else dim // 2
            inter_ch = min(inter_ch, dim)
            if stage_idx < len(self.embed_dims) - 1:
                block = NonLocalBlock2d(
                    in_channels=dim,
                    inter_channels=inter_ch,
                    bn_layer=nl_bn_layer,
                )
            else:
                block = NonLocalBlockLinear(
                    in_channels=dim,
                    inter_channels=inter_ch,
                    bn_layer=nl_bn_layer,
                )

            self.nonlocal_blocks.append(block)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Non-Local blocks inserted after each of {len(self.embed_dims)} stages")
        _logger.info(f"Non-Local inter_channels: {[blk.inter_channels for blk in self.nonlocal_blocks]}")

    def load_state_dict(self, state_dict, strict):
        for i, nl_block in enumerate(self.nonlocal_blocks):
            for name, param in nl_block.state_dict().items():
                key = f"nonlocal_blocks.{i}.{name}"
                if key not in state_dict:
                    state_dict[key] = param.clone()

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)

            if isinstance(block, nn.Sequential):
                x = self.nonlocal_blocks[stage_idx](x)
                stage_idx += 1

            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))

        if self.fork_feat:
            return outs
        return x


###### Multi-Head Self-Attention (MHSA) variant
class FastViT_mhsa(FastViT_Projector):
    """FastViT with MultiHeadNonLocalBlock2d (MHSA) after every stage."""

    def __init__(
        self,
        freeze_backbone=True,
        clip_dim=768,
        mhsa_inter_channels=None,
        mhsa_num_heads=4,
        mhsa_bn_layer=True,
        **kwargs,
    ):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        # Create MHSA blocks after every stage
        self.mhsa_blocks = nn.ModuleList()
        for dim in self.embed_dims:
            inter_ch = mhsa_inter_channels if mhsa_inter_channels is not None else dim // 2
            inter_ch = min(inter_ch, dim)
            # Ensure inter_ch is divisible by num_heads
            inter_ch = (inter_ch // mhsa_num_heads) * mhsa_num_heads
            inter_ch = max(inter_ch, mhsa_num_heads)
            block = MultiHeadNonLocalBlock2d(
                in_channels=dim,
                inter_channels=inter_ch,
                num_heads=mhsa_num_heads,
                bn_layer=mhsa_bn_layer,
            )
            self.mhsa_blocks.append(block)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"MHSA blocks inserted after each of {len(self.embed_dims)} stages")
        _logger.info(f"MHSA inter_channels: {[blk.inter_channels for blk in self.mhsa_blocks]}")
        _logger.info(f"MHSA num_heads: {mhsa_num_heads}")

    def load_state_dict(self, state_dict, strict):
        for i, mhsa_block in enumerate(self.mhsa_blocks):
            for name, param in mhsa_block.state_dict().items():
                key = f"mhsa_blocks.{i}.{name}"
                if key not in state_dict:
                    state_dict[key] = param.clone()

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)

            if isinstance(block, nn.Sequential):
                x = self.mhsa_blocks[stage_idx](x)
                stage_idx += 1

            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))

        if self.fork_feat:
            return outs
        return x


###### Neck Probe (for evaluation with forward_backbone / forward_classification_neck)
class FastViT_NeckProbe(FastViT):
    """Minimal wrapper around FastViT for neck/backbone feature extraction.
    
    Inherits forward_backbone() and forward_classification_neck() from FastViT.
    No extra layers added - just allows proper model registration and loading.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


#### Non-Local
@register_model
def fastvit_sa36_nonlocal(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with Non-Local blocks."""
    model = FastViT_nonlocal(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model


@register_model
def fastvit_sa36_mhsa(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant with MHSA blocks."""
    model = FastViT_mhsa(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model


@register_model
def fastvit_sa36_neckprobe(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 for neck/backbone probing."""
    model = FastViT_NeckProbe(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model
