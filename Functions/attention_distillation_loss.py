"""
Attention Distillation Loss for CLIP-to-FastViT Knowledge Transfer.

This module provides loss functions for attention-based knowledge distillation
between CLIP's vision transformer and FastViT's attention blocks.

The main approach:
1. Extract attention maps from both teacher (CLIP) and student (FastViT)
2. Take the last layer's attention from each block
3. Align spatial dimensions
4. Compute MSE loss to encourage student to match teacher attention

Reference:
- Original CLIP: Radford et al. (2021)
- FastViT: Han et al. (2023)
"""

import torch
import torch.nn as nn
from .attention_utils import attention_distillation_loss


class AttentionDistillationLoss(nn.Module):
    """
    Loss module for attention-based knowledge distillation.

    Takes the last attention layer from each block and computes KL divergence loss
    after spatial alignment. KL divergence is preferred over MSE because attention
    maps are probability distributions (softmax output).

    Usage:
        loss_fn = AttentionDistillationLoss(loss_type='kl')
        teacher_attn = [...]  # List of attention layers from teacher
        student_attn = [...]  # List of attention layers from student
        loss = loss_fn(teacher_attn, student_attn)
    """

    def __init__(self, loss_type='kl', normalize=False, reduction='mean'):
        """
        Initialize attention distillation loss.

        Args:
            loss_type: Type of loss to use. Options: 'kl', 'mse', 'mse_logits'.
                      Default: 'kl' (better for probability distributions)
            normalize: If True, apply row-wise normalization before loss.
                      Default: False (KL divergence doesn't need pre-normalization)
            reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.
                      Default: 'mean' (gives normalized loss per spatial location)
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.reduction = reduction

        if loss_type.lower() not in ['kl', 'mse', 'mse_logits']:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'kl', 'mse', or 'mse_logits'")

    def forward(self, teacher_attn_layers, student_attn_layers,
                spatial_align=True, teacher_logits_layers=None, student_logits_layers=None):
        """
        Compute attention distillation loss using last layer from each block.

        Args:
            teacher_attn_layers: List of teacher attention tensors,
                                each [B, num_heads, N_teacher, N_teacher]
                                For CLIP layer4: 2 layers of [B, 16, 257, 257]
                                Uses the last layer: [B, 16, 257, 257]
            student_attn_layers: List of student attention tensors,
                                each [B, num_heads, N_student, N_student]
                                For FastViT stage4: 6 layers of [B, 16, 49, 49]
                                Uses the last layer: [B, 16, 49, 49]
            spatial_align: If True, align teacher spatial dimensions to student.
                          Default: True
            teacher_logits_layers: Optional list of teacher attention logits (raw, before softmax).
                                  Required if loss_type='mse_logits'.
            student_logits_layers: Optional list of student attention logits (raw, before softmax).
                                  Required if loss_type='mse_logits'.

        Returns:
            loss: Scalar tensor containing the distillation loss
        """
        if not teacher_attn_layers:
            raise ValueError(
                "teacher_attn_layers is empty. "
                "Check that CLIP attention extraction is working correctly. "
                "Ensure get_layer4_attention_maps() is returning attention maps."
            )
        if not student_attn_layers:
            raise ValueError(
                "student_attn_layers is empty. "
                "Check that FastViT attention extraction is working correctly. "
                "Ensure AttentionMapExtractor hooks are registering and capturing attention. "
                "Debug info: Check if stage4_layer_names match the actual module names in the model."
            )

        # Take the last layer from each block
        if isinstance(teacher_attn_layers, list):
            teacher_attn = teacher_attn_layers[-1]
        else:
            teacher_attn = teacher_attn_layers

        if isinstance(student_attn_layers, list):
            student_attn = student_attn_layers[-1]
        else:
            student_attn = student_attn_layers

        # Extract logits if provided
        teacher_logits = None
        student_logits = None

        if teacher_logits_layers and student_logits_layers:
            if isinstance(teacher_logits_layers, list):
                teacher_logits = teacher_logits_layers[-1]
            else:
                teacher_logits = teacher_logits_layers

            if isinstance(student_logits_layers, list):
                student_logits = student_logits_layers[-1]
            else:
                student_logits = student_logits_layers

        # Compute distillation loss
        loss = attention_distillation_loss(
            teacher_attn,
            student_attn,
            spatial_align=spatial_align,
            normalize=self.normalize,
            reduction=self.reduction,
            loss_type=self.loss_type,
            teacher_logits=teacher_logits,
            student_logits=student_logits
        )

        return loss
