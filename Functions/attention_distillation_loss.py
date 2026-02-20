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

    Takes the last attention layer from each block and computes MSE loss
    after spatial alignment.

    Usage:
        loss_fn = AttentionDistillationLoss(normalize=True)
        teacher_attn = [...]  # List of attention layers from teacher
        student_attn = [...]  # List of attention layers from student
        loss = loss_fn(teacher_attn, student_attn)
    """

    def __init__(self, normalize=True, reduction='mean'):
        """
        Initialize attention distillation loss.

        Args:
            normalize: If True, normalize attention maps before computing loss.
                      Default: True
            reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.
                      Default: 'mean'
        """
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, teacher_attn_layers, student_attn_layers,
                spatial_align=True):
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

        # Compute distillation loss
        loss = attention_distillation_loss(
            teacher_attn,
            student_attn,
            spatial_align=spatial_align,
            normalize=self.normalize,
            reduction=self.reduction
        )

        return loss


# Example usage
if __name__ == '__main__':
    print("Testing AttentionDistillationLoss...")

    # Create loss module
    loss_fn = AttentionDistillationLoss(normalize=True)

    # Simulate CLIP layer4 attention (2 layers, uses last)
    teacher_layers = [torch.randn(2, 16, 257, 257) for _ in range(2)]

    # Simulate FastViT stage4 attention (6 layers, uses last)
    student_layers = [torch.randn(2, 16, 49, 49) for _ in range(6)]

    # Compute loss
    loss = loss_fn(teacher_layers, student_layers)

    print(f"Teacher input: 2 layers, using last: {teacher_layers[-1].shape}")
    print(f"Student input: 6 layers, using last: {student_layers[-1].shape}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires grad: {loss.requires_grad}")

    # Test backward pass
    loss.backward()
    print("Backward pass successful!")
    print("Loss can be used in optimization!")
