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
        nl_grad_scale=0.1,
        nl_init_gate=0.0,
        **kwargs,
    ):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        # Unfreeze the classification head so it can adapt to
        # feature changes introduced by the non-local blocks.
        # Without this, the frozen head fights the non-local blocks
        # via the CE loss and destroys the learned features.
        if freeze_backbone:
            for name, param in self.head.named_parameters():
                param.requires_grad = True
                _logger.info(f"  Unfreezing head param: head.{name}")

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
                    name=f"NonLocal_Stage_{stage_idx}",
                )
            else:
                block = NonLocalBlockLinear(
                    in_channels=dim,
                    inter_channels=inter_ch,
                    bn_layer=nl_bn_layer,
                    name=f"NonLocal_Stage_{stage_idx}",
                )

            self.nonlocal_blocks.append(block)

        # Learnable gate scalars (one per stage) that control how much
        # the non-local output contributes. Initialised near zero so the
        # model starts close to the pre-trained identity mapping and
        # gradually lets non-local features bleed in as training proceeds.
        self.nl_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(nl_init_gate))
            for _ in range(len(self.embed_dims))
        ])

        # When the backbone is frozen, scale down the gradients flowing
        # into non-local params so they update ~10x slower than the
        # projector / head.  This prevents the non-local blocks from
        # destabilising the intermediate features of the frozen backbone.
        self._nl_grad_scale = nl_grad_scale
        if freeze_backbone and nl_grad_scale < 1.0:
            self._register_nl_gradient_scaling(nl_grad_scale)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"Non-Local blocks inserted after each of {len(self.embed_dims)} stages")
        _logger.info(f"Non-Local inter_channels: {[blk.inter_channels for blk in self.nonlocal_blocks]}")
        _logger.info(f"Non-Local grad_scale: {nl_grad_scale}, init_gate: {nl_init_gate}")

    # ------------------------------------------------------------------
    def _register_nl_gradient_scaling(self, scale: float):
        """Register backward hooks that multiply gradients by *scale*,
        giving non-local parameters a lower effective learning rate."""
        for block in self.nonlocal_blocks:
            for param in block.parameters():
                if param.requires_grad:
                    param.register_hook(lambda grad, s=scale: grad * s)
        for gate_param in self.nl_gates:
            gate_param.register_hook(lambda grad, s=scale: grad * s)
        _logger.info(f"  Registered gradient scaling ({scale}) on non-local params")

    def load_state_dict(self, state_dict, strict):
        for i, nl_block in enumerate(self.nonlocal_blocks):
            for name, param in nl_block.state_dict().items():
                key = f"nonlocal_blocks.{i}.{name}"
                if key not in state_dict:
                    state_dict[key] = param.clone()
        # Inject default gate values when loading older checkpoints
        for i, gate in enumerate(self.nl_gates):
            key = f"nl_gates.{i}"
            if key not in state_dict:
                state_dict[key] = gate.data.clone()

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)

            if isinstance(block, nn.Sequential):
                # Apply gated non-local block: x + gate * (NL(x) - x)
                # NL(x) already includes the residual, so NL(x) - x = non-local contrib only.
                identity = x
                nl_out = self.nonlocal_blocks[stage_idx](x)
                gate = self.nl_gates[stage_idx].sigmoid()  # squash to [0, 1]
                x = identity + gate * (nl_out - identity)
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
        mhsa_num_heads=5,
        mhsa_bn_layer=True,
        mhsa_grad_scale=0.1,
        mhsa_init_gate=0.0,
        **kwargs,
    ):
        super().__init__(freeze_backbone=freeze_backbone, clip_dim=clip_dim, **kwargs)

        # Unfreeze the classification head so it can adapt to
        # feature changes introduced by the MHSA blocks.
        if freeze_backbone:
            for name, param in self.head.named_parameters():
                param.requires_grad = True
                _logger.info(f"  Unfreezing head param: head.{name}")

        # Create MHSA blocks after every stage
        self.mhsa_blocks = nn.ModuleList()
        for stage_idx, dim in enumerate(self.embed_dims):
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
                name=f"MHSA_Stage_{stage_idx}",
            )
            self.mhsa_blocks.append(block)

        # Learnable gate scalars — same idea as non-local variant
        self.mhsa_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(mhsa_init_gate))
            for _ in range(len(self.embed_dims))
        ])

        # Gradient scaling for frozen backbone training
        self._mhsa_grad_scale = mhsa_grad_scale
        if freeze_backbone and mhsa_grad_scale < 1.0:
            self._register_mhsa_gradient_scaling(mhsa_grad_scale)

        num_sequential = sum(isinstance(m, nn.Sequential) for m in self.network)
        _logger.info(f"Number of nn.Sequential blocks in self.network: {num_sequential}")
        _logger.info(f"MHSA blocks inserted after each of {len(self.embed_dims)} stages")
        _logger.info(f"MHSA inter_channels: {[blk.inter_channels for blk in self.mhsa_blocks]}")
        _logger.info(f"MHSA num_heads: {mhsa_num_heads}")
        _logger.info(f"MHSA grad_scale: {mhsa_grad_scale}, init_gate: {mhsa_init_gate}")

    def _register_mhsa_gradient_scaling(self, scale: float):
        for block in self.mhsa_blocks:
            for param in block.parameters():
                if param.requires_grad:
                    param.register_hook(lambda grad, s=scale: grad * s)
        for gate_param in self.mhsa_gates:
            gate_param.register_hook(lambda grad, s=scale: grad * s)
        _logger.info(f"  Registered gradient scaling ({scale}) on MHSA params")

    def load_state_dict(self, state_dict, strict):
        for i, mhsa_block in enumerate(self.mhsa_blocks):
            for name, param in mhsa_block.state_dict().items():
                key = f"mhsa_blocks.{i}.{name}"
                if key not in state_dict:
                    state_dict[key] = param.clone()
        for i, gate in enumerate(self.mhsa_gates):
            key = f"mhsa_gates.{i}"
            if key not in state_dict:
                state_dict[key] = gate.data.clone()

        super().load_state_dict(state_dict, strict)

    def forward_tokens(self, x: torch.Tensor):
        stage_idx = 0
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)

            if isinstance(block, nn.Sequential):
                identity = x
                mhsa_out = self.mhsa_blocks[stage_idx](x)
                gate = self.mhsa_gates[stage_idx].sigmoid()
                x = identity + gate * (mhsa_out - identity)
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
    kwargs.setdefault('mhsa_num_heads', 5)
    model = FastViT_mhsa(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model


@register_model
def fastvit_sa36_neckprobe(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 for neck/backbone probing."""
    model = FastViT_NeckProbe(**fastvit_sa36_config, **kwargs)
    model.default_cfg = default_cfgs["fastvit_m"]
    return model
