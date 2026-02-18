import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple, List


class LossManager:
    """
    Composable loss manager that computes only registered losses.
    Losses are added via add_loss() and computed in compute().
    """

    def __init__(self, base_loss_fn: nn.Module):
        self.base_loss_fn = base_loss_fn
        self._losses: List[Tuple[str, Callable]] = []

    def add_loss(self, name: str, loss_fn: Callable) -> "LossManager":
        """Register a loss function. Returns self for chaining."""
        self._losses.append((name, loss_fn))
        return self

    def compute(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute base loss + all registered losses."""
        loss_dict = {}

        base_loss = self.base_loss_fn(output, target)
        total_loss = base_loss
        loss_dict["Base Loss"] = base_loss

        for name, loss_fn in self._losses:
            loss_val = loss_fn(
                output=output,
                target=target,
                projected_embed=projected_embed,
                clip_image_features=clip_image_features,
                logit_scale=logit_scale,
            )
            if loss_val is not None:
                total_loss = total_loss + loss_val
                loss_dict[name] = loss_val

        return total_loss, loss_dict


# =============================================================================
# Loss computation functions
# =============================================================================


def create_clip_loss(
    clip_loss_fn: Callable,
    clip_text_features: torch.Tensor,
    clip_logit_scale: Optional[torch.Tensor] = None,
) -> Callable:
    """Create a CLIP loss function with pre-bound text features.
    If clip_logit_scale is provided it is used as a static fallback when no
    trainable scale is supplied at compute time."""

    def compute_clip_loss(
        output: torch.Tensor,  # noqa: ARG001
        target: torch.Tensor,
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,  # noqa: ARG001
        logit_scale: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if projected_embed is None:
            return None

        feats = projected_embed[0] if isinstance(projected_embed, (tuple, list)) else projected_embed
        if feats.ndim == 4 and feats.shape[2] > 1:
            feats = feats.mean(dim=[2, 3])

        scale = logit_scale if logit_scale is not None else clip_logit_scale
        return clip_loss_fn(feats, clip_text_features[target], scale)

    return compute_clip_loss


def create_mse_loss() -> Callable:
    """Create an MSE loss function for distillation."""
    mse_fn = nn.MSELoss()

    def compute_mse_loss(
        output: torch.Tensor,  # noqa: ARG001
        target: torch.Tensor,  # noqa: ARG001
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,  # noqa: ARG001
    ) -> Optional[torch.Tensor]:
        if projected_embed is None or clip_image_features is None:
            return None

        feats = projected_embed[0] if isinstance(projected_embed, (tuple, list)) else projected_embed
        if feats.ndim == 4 and feats.shape[2] > 1:
            feats = feats.mean(dim=[2, 3])

        feats = F.normalize(feats, dim=-1)
        clip_image_features = F.normalize(clip_image_features.float(), dim=-1)

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
    clip_logit_scale: Optional[torch.Tensor] = None,
) -> LossManager:
    """
    Create a LossManager for standard baseline training.
    Losses: base_loss, clip_loss
    """
    manager = LossManager(base_loss_fn)

    if clip_loss_fn is not None and clip_text_features is not None:
        manager.add_loss(
            "CLIP Loss",
            create_clip_loss(clip_loss_fn, clip_text_features, clip_logit_scale),
        )

    return manager


def create_distillation_loss_manager(
    base_loss_fn: nn.Module,
    clip_loss_fn: Optional[Callable] = None,
    clip_text_features: Optional[torch.Tensor] = None,
    clip_logit_scale: Optional[torch.Tensor] = None,
) -> LossManager:
    """
    Create a LossManager for distillation training.
    Losses: base_loss, clip_loss, mse_loss
    """
    manager = LossManager(base_loss_fn)

    if clip_loss_fn is not None and clip_text_features is not None:
        manager.add_loss(
            "CLIP Loss",
            create_clip_loss(clip_loss_fn, clip_text_features, clip_logit_scale),
        )

    manager.add_loss("MSE Loss", create_mse_loss())

    return manager


def get_loss_manager_for_method(
    method: str,
    base_loss_fn: nn.Module,
    clip_loss_fn: Optional[Callable] = None,
    clip_text_features: Optional[torch.Tensor] = None,
    clip_logit_scale: Optional[torch.Tensor] = None,
    ) -> LossManager:
    """
    Factory function to create appropriate LossManager based on training method.

    Args:
        method: Training method name. Supported: 'default', 'baseline', 'distillation'
        base_loss_fn: Base classification loss function
        clip_loss_fn: Optional CLIP loss function
        clip_text_features: Optional pre-computed CLIP text features
        clip_logit_scale: Optional static CLIP logit scale (overridden by model's
                          trainable log_logit_scale when present)

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
            clip_logit_scale=clip_logit_scale,
        )
    elif method == "distillation":
        return create_distillation_loss_manager(
            base_loss_fn=base_loss_fn,
            clip_loss_fn=clip_loss_fn,
            clip_text_features=clip_text_features,
            clip_logit_scale=clip_logit_scale,
        )
    else:
        raise ValueError(
            f"Unknown training method: '{method}'. "
            f"Supported methods: 'default', 'baseline', 'distillation'"
        )