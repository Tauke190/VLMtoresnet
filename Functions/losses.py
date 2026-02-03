import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple


class LossManager:
    def __init__(
        self,
        args,
        base_loss_fn: nn.Module,
        clip_loss_fn: Optional[Callable] = None,
        clip_text_features: Optional[torch.Tensor] = None,
        clip_logit_scale: Optional[torch.Tensor] = None,
    ):
        self.base_loss_fn = base_loss_fn
        self.clip_loss_fn = clip_loss_fn
        self.clip_text_features = clip_text_features
        self.clip_logit_scale = clip_logit_scale
        self.clip_loss_weight = getattr(args, 'clip_loss_weight', 0.0)
        self.mse_loss_weight = getattr(args, 'mse_loss_weight', 0.0)
        self.mse_loss_fn = nn.MSELoss()

    def compute(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        projected_embed: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = {}  # stores tensors, not floats - call .item() only when logging

        base_loss = self.base_loss_fn(output, target)
        total_loss = base_loss
        loss_dict['Base Loss'] = base_loss

        if self.clip_loss_weight > 0.0 and projected_embed is not None:
            clip_loss = self._compute_clip_loss(projected_embed, target)
            total_loss = total_loss + self.clip_loss_weight * clip_loss
            loss_dict['CLIP loss'] = clip_loss

        if self.mse_loss_weight > 0.0 and projected_embed is not None and clip_image_features is not None:
            mse_loss = self._compute_mse_loss(projected_embed, clip_image_features)
            total_loss = total_loss + self.mse_loss_weight * mse_loss
            loss_dict['MSE loss'] = mse_loss

        return total_loss, loss_dict

    def _compute_clip_loss(self, projected_embed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.clip_loss_fn is None or self.clip_text_features is None:
            return torch.tensor(0.0, device=projected_embed.device)

        feats = projected_embed[0] if isinstance(projected_embed, (tuple, list)) else projected_embed
        if feats.ndim == 4 and feats.shape[2] > 1:
            feats = feats.mean(dim=[2, 3])

        return self.clip_loss_fn(feats, self.clip_text_features[target], self.clip_logit_scale)

    def _compute_mse_loss(self, projected_embed: torch.Tensor, clip_image_features: torch.Tensor) -> torch.Tensor:
        feats = projected_embed[0] if isinstance(projected_embed, (tuple, list)) else projected_embed
        if feats.ndim == 4 and feats.shape[2] > 1:
            feats = feats.mean(dim=[2, 3])

        feats = F.normalize(feats, dim=-1)
        clip_image_features = F.normalize(clip_image_features.float(), dim=-1)

        return self.mse_loss_fn(feats, clip_image_features)