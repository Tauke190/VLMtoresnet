import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple, List


class LossManager:
    """
    Composable loss manager that computes only registered losses with optional weights.
    Losses are added via add_loss() and computed in compute().
    """

    def __init__(self, base_loss_fn: nn.Module):
        self.base_loss_fn = base_loss_fn
        self._losses: List[Tuple[str, Callable, float]] = []

    def add_loss(self, name: str, loss_fn: Callable, weight: float = 1.0) -> "LossManager":
        """Register a loss function with optional weight. Returns self for chaining."""
        self._losses.append((name, loss_fn, weight))
        return self

    def compute(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute base loss + all registered losses with their respective weights."""
        loss_dict = {}

        # Base loss is always computed
        base_loss = self.base_loss_fn(output, target)
        total_loss = base_loss
        loss_dict["Base Loss"] = base_loss

        # Compute each registered loss
        for name, loss_fn, weight in self._losses:
            loss_val = loss_fn(
                output=output,
                target=target,
                projected_embed=projected_embed,
                clip_image_features=clip_image_features,
                logit_scale=logit_scale,
                **kwargs,
            )
            if loss_val is not None:
                weighted_loss = weight * loss_val
                total_loss = total_loss + weighted_loss
                loss_dict[name] = weighted_loss  # Store weighted loss for logging

        return total_loss, loss_dict


# =============================================================================
# Loss computation functions
# =============================================================================

def create_attn_distill_loss(attn_distill_weight: float = 1.0) -> Callable:
    """Create an attention distillation loss using last-layer spatial alignment."""
    from .attention_distillation_loss import AttentionDistillationLoss

    attn_loss_fn = AttentionDistillationLoss(normalize=True)

    def compute_attn_distill_loss(
        output: torch.Tensor,  # noqa: ARG001
        target: torch.Tensor,  # noqa: ARG001
        projected_embed: Optional[torch.Tensor] = None,  # noqa: ARG001
        clip_image_features: Optional[torch.Tensor] = None,  # noqa: ARG001
        teacher_attn_layers=None,
        student_attn_layers=None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if teacher_attn_layers is None or student_attn_layers is None:
            return None
        return attn_distill_weight * attn_loss_fn(teacher_attn_layers, student_attn_layers)

    return compute_attn_distill_loss

def create_clip_loss(
    clip_loss_fn: Callable,
    clip_text_features: torch.Tensor,
) -> Callable:
    """Create a CLIP loss function with pre-bound text features.

    logit_scale MUST be explicitly passed during compute() from the model's forward pass.
    """

    def compute_clip_loss(
        output: torch.Tensor,  # noqa: ARG001
        target: torch.Tensor,
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,  # noqa: ARG001
        logit_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if projected_embed is None:
            return None

        feats = projected_embed[0] if isinstance(projected_embed, (tuple, list)) else projected_embed
        if feats.ndim == 4 and feats.shape[2] > 1:
            feats = feats.mean(dim=[2, 3])

        # logit_scale MUST be provided from model forward pass
        assert logit_scale is not None, "logit_scale must be explicitly passed from model"
        return clip_loss_fn(feats, clip_text_features[target], logit_scale)

    return compute_clip_loss


def create_mse_loss() -> Callable:
    """Create an MSE loss function for distillation."""
    mse_fn = nn.MSELoss()

    def compute_mse_loss(
        output: torch.Tensor,  # noqa: ARG001
        target: torch.Tensor,  # noqa: ARG001
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if projected_embed is None or clip_image_features is None:
            return None

        feats = projected_embed[0] if isinstance(projected_embed, (tuple, list)) else projected_embed
        if feats.ndim == 4 and feats.shape[2] > 1:
            feats = feats.mean(dim=[2, 3])

        feats = F.normalize(feats, dim=-1)

        clip_image_features = clip_image_features.float() # Unormalized CLIP features may be in fp16, ensure they're float for MSE

        return mse_fn(feats, clip_image_features)

    return compute_mse_loss


# =============================================================================
# Factory functions for different training methods
# =============================================================================


def create_default_loss_manager(base_loss_fn: nn.Module) -> LossManager:
    """
    Create a LossManager with base loss only for vanilla fastvit
    Losses: base_loss
    """
    return LossManager(base_loss_fn)


def create_baseline_loss_manager(
    base_loss_fn: nn.Module,
    clip_loss_fn: Optional[Callable] = None,
    clip_text_features: Optional[torch.Tensor] = None,
) -> LossManager:
    """
    Create a LossManager for standard baseline training.
    Losses: base_loss, clip_loss
    """
    manager = LossManager(base_loss_fn)

    if clip_loss_fn is not None and clip_text_features is not None:
        manager.add_loss(
            "CLIP Loss",
            create_clip_loss(clip_loss_fn, clip_text_features),
        )

    return manager


def create_distillation_loss_manager(
    base_loss_fn: nn.Module,
    clip_loss_fn: Optional[Callable] = None,
    clip_text_features: Optional[torch.Tensor] = None,
    mse_distill_weight: float = 1.0,
) -> LossManager:
    """
    Create a LossManager for distillation training.
    Losses: base_loss, clip_loss, mse_loss

    Args:
        base_loss_fn: Base classification loss
        clip_loss_fn: Optional CLIP contrastive loss
        clip_text_features: Optional CLIP text features
        mse_distill_weight: Weight for MSE distillation loss. Use lower values (e.g., 0.1-0.5)
                           if MSE loss is too strong and inhibits learning.
    """
    manager = LossManager(base_loss_fn)

    if clip_loss_fn is not None and clip_text_features is not None:
        manager.add_loss(
            "CLIP Loss",
            create_clip_loss(clip_loss_fn, clip_text_features),
        )

    manager.add_loss("MSE Loss", create_mse_loss(), weight=mse_distill_weight)

    return manager


def create_attention_distillation_loss_manager(
    base_loss_fn: nn.Module,
    clip_loss_fn: Optional[Callable] = None,
    clip_text_features: Optional[torch.Tensor] = None,
    attn_distill_weight: float = 1.0,
) -> LossManager:
    """
    Create a LossManager for attention distillation training.
    Losses: base_loss, clip_loss, attention_distillation_loss
    """
    manager = LossManager(base_loss_fn)

    if clip_loss_fn is not None and clip_text_features is not None:
        manager.add_loss(
            "CLIP Loss",
            create_clip_loss(clip_loss_fn, clip_text_features),
        )

    manager.add_loss("Attn Distill Loss", create_attn_distill_loss(attn_distill_weight))

    return manager


def get_loss_manager_for_method(
    method: str,
    base_loss_fn: nn.Module,
    clip_loss_fn: Optional[Callable] = None,
    clip_text_features: Optional[torch.Tensor] = None,
    attn_distill_weight: float = 1.0,
    mse_distill_weight: float = 1.0,
    ) -> LossManager:
    """
    Factory function to create appropriate LossManager based on training method.

    Args:
        method: Training method name. Supported: 'default', 'baseline', 'distillation', 'attention_distillation'
        base_loss_fn: Base classification loss function
        clip_loss_fn: Optional CLIP loss function
        clip_text_features: Optional pre-computed CLIP text features
        attn_distill_weight: Weight for attention distillation loss
        mse_distill_weight: Weight for MSE distillation loss (used with 'distillation' method)

    Returns:
        Configured LossManager instance
    """
    method = method.lower()

    if method == "default":
        return create_default_loss_manager(base_loss_fn=base_loss_fn)
    elif method == "baseline":
        return create_baseline_loss_manager(
            base_loss_fn=base_loss_fn,
            clip_loss_fn=clip_loss_fn,
            clip_text_features=clip_text_features,
        )
    elif method == "distillation":
        return create_distillation_loss_manager(
            base_loss_fn=base_loss_fn,
            clip_loss_fn=clip_loss_fn,
            clip_text_features=clip_text_features,
            mse_distill_weight=mse_distill_weight,
        )
    elif method == "attention_distillation":
        return create_attention_distillation_loss_manager(
            base_loss_fn=base_loss_fn,
            clip_loss_fn=clip_loss_fn,
            clip_text_features=clip_text_features,
            attn_distill_weight=attn_distill_weight,
        )
    else:
        raise ValueError(
            f"Unknown training method: '{method}'. "
            f"Supported methods: 'default', 'baseline', 'distillation', 'attention_distillation'"
        )