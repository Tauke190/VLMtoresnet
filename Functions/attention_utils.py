"""
Attention Map Utilities for Knowledge Distillation.

This module provides utilities for:
- Spatial alignment of attention maps between different architectures
- Attention distillation loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_attention_spatial(clip_attn, target_size=7):
    """
    Align CLIP's 16×16 patch attention to FastViT's 7×7 spatial size.

    CLIP ViT produces attention over 257 tokens (1 CLS + 256 patches in 16×16 grid).
    FastViT produces attention over 49 tokens (7×7 grid, no CLS).

    This function:
    1. Removes CLS token from CLIP attention
    2. Reshapes to spatial dimensions [16, 16, 16, 16] (query_H, query_W, key_H, key_W)
    3. Applies adaptive pooling to downsample to target size

    Args:
        clip_attn: CLIP attention [B, num_heads, 257, 257] or [B, 257, 257] (with CLS token)
                  3D tensors are converted to 4D by adding a head dimension.
        target_size: Target spatial dimension (default: 7 for 7×7 grid)

    Returns:
        aligned: Spatially aligned attention [B, num_heads, target_size^2, target_size^2]
                 or [B, target_size^2, target_size^2] if input was 3D

    Example:
        >>> clip_attn = torch.randn(2, 16, 257, 257)
        >>> aligned = align_attention_spatial(clip_attn, target_size=7)
        >>> aligned.shape
        torch.Size([2, 16, 49, 49])
    """
    # Handle 3D tensors (B, N, N) by adding head dimension
    is_3d_input = clip_attn.dim() == 3
    if is_3d_input:
        clip_attn = clip_attn.unsqueeze(1)  # [B, 257, 257] -> [B, 1, 257, 257]

    B, num_heads = clip_attn.shape[:2]

    # Step 1: Remove CLS token (index 0)
    # Original: [B, H, 257, 257] where 257 = 1 CLS + 256 patches
    # Remove row 0 (CLS as query) and column 0 (CLS as key)
    patch_attn = clip_attn[:, :, 1:, 1:]  # [B, H, 256, 256]

    # Step 2: Reshape to spatial dimensions
    # 256 tokens = 16×16 spatial grid
    # Attention is [query_tokens, key_tokens] = [16*16, 16*16]
    # Reshape to [query_H, query_W, key_H, key_W]
    spatial_size = 16  # sqrt(256) = 16
    spatial_attn = patch_attn.reshape(B, num_heads, spatial_size, spatial_size,
                                      spatial_size, spatial_size)
    # Shape: [B, H, 16, 16, 16, 16]

    # Step 3: Downsample both query and key dimensions
    # We need to pool from [16, 16, 16, 16] to [7, 7, 7, 7]

    # Pool query dimension: [B, H, 16, 16, 16, 16] -> [B, H, 7, 7, 16, 16]
    spatial_attn = spatial_attn.permute(0, 1, 4, 5, 2, 3)  # [B, H, 16, 16, 16, 16]
    spatial_attn = spatial_attn.reshape(B * num_heads * spatial_size * spatial_size,
                                       spatial_size, spatial_size)
    spatial_attn = F.adaptive_avg_pool2d(spatial_attn, (target_size, target_size))
    spatial_attn = spatial_attn.reshape(B, num_heads, spatial_size, spatial_size,
                                       target_size, target_size)

    # Pool key dimension: [B, H, 16, 16, 7, 7] -> [B, H, 7, 7, 7, 7]
    spatial_attn = spatial_attn.permute(0, 1, 4, 5, 2, 3)  # [B, H, 7, 7, 16, 16]
    spatial_attn = spatial_attn.reshape(B * num_heads * target_size * target_size,
                                       spatial_size, spatial_size)
    spatial_attn = F.adaptive_avg_pool2d(spatial_attn, (target_size, target_size))
    spatial_attn = spatial_attn.reshape(B, num_heads, target_size, target_size,
                                       target_size, target_size)

    # Step 4: Flatten back to [B, H, target_size^2, target_size^2]
    aligned = spatial_attn.reshape(B, num_heads, target_size * target_size,
                                   target_size * target_size)

    # If input was 3D, remove the added head dimension
    if is_3d_input:
        aligned = aligned.squeeze(1)  # [B, 1, 49, 49] -> [B, 49, 49]

    return aligned


def attention_distillation_loss(teacher_attn, student_attn,
                                spatial_align=True,
                                reduction='mean', loss_type='kl',
                                teacher_logits=None, student_logits=None):
    """
    Compute distillation loss between teacher and student attention maps.

    This loss encourages the student network to learn similar attention
    patterns to the teacher network.

    Args:
        teacher_attn: Teacher attention (softmax) — must be valid probability distribution
                     - CLIP (head-averaged): [B, 257, 257]
                     - Or with heads: [B, 16, 257, 257]
        student_attn: Student attention (softmax) — must be valid probability distribution
                     - FastViT (with heads): [B, 16, 49, 49]
                     - Or head-averaged: [B, 49, 49]
        spatial_align: If True, spatially align teacher to student dimensions
                      and average across heads if needed. Default: True
        normalize: If True, apply row-wise normalization (NOT recommended with KL).
                  Default: False (KL divergence doesn't need it)
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
        loss_type: Loss function to use. Options:
                  - 'kl': KL divergence on softmax attention (recommended)
                  - 'mse': MSE loss on softmax attention
                  - 'mse_logits': MSE loss on raw logits (before softmax)
                  Default: 'kl'
        teacher_logits: Teacher attention logits (raw, before softmax).
                       Required if loss_type='mse_logits'.
                       Shape: [B, num_heads, N, N]
        student_logits: Student attention logits (raw, before softmax).
                       Required if loss_type='mse_logits'.
                       Shape: [B, num_heads, N, N]

    Returns:
        loss: Scalar tensor (if reduction='mean' or 'sum')
              or [B] tensor (if reduction='none')

    Example:
        >>> # CLIP (head-averaged) to FastViT (with heads) - KL divergence
        >>> teacher = torch.randn(2, 257, 257).softmax(dim=-1)
        >>> student = torch.randn(2, 16, 49, 49).softmax(dim=-1)
        >>> loss = attention_distillation_loss(teacher, student, loss_type='kl')
        >>> loss.backward()

        >>> # MSE on raw logits (captures attention sharpness)
        >>> teacher_logits = torch.randn(2, 16, 257, 257)
        >>> student_logits = torch.randn(2, 16, 49, 49)
        >>> loss = attention_distillation_loss(
        ...     teacher, student,
        ...     loss_type='mse_logits',
        ...     teacher_logits=teacher_logits,
        ...     student_logits=student_logits
        ... )
    """
    # Ensure tensors are on the same device (student attention might be on CPU)
    if student_attn.device != teacher_attn.device:
        student_attn = student_attn.to(teacher_attn.device)

    # Handle spatial alignment
    if spatial_align:
        # Check if teacher needs spatial alignment
        if teacher_attn.shape[-1] == 257:  # CLIP with CLS token
            # Determine target size from student
            student_n = student_attn.shape[-1]
            target_spatial = int(student_n ** 0.5)
            if target_spatial * target_spatial != student_n:
                raise ValueError(f"Student attention size {student_n} is not a perfect square")

            teacher_attn = align_attention_spatial(teacher_attn, target_size=target_spatial)

        # Handle head dimension mismatch
        # CLIP attention is usually head-averaged [B, N, N]
        # FastViT attention has heads [B, num_heads, N, N]
        if teacher_attn.dim() == 3 and student_attn.dim() == 4:
            # Teacher has no heads, student has heads - average student's heads
            student_attn = student_attn.mean(dim=1)  # [B, 16, N, N] -> [B, N, N]
        elif teacher_attn.dim() == 4 and student_attn.dim() == 3:
            # Teacher has heads, student doesn't - average teacher's heads
            teacher_attn = teacher_attn.mean(dim=1)  # [B, 16, N, N] -> [B, N, N]

        # Ensure dimensions match
        if teacher_attn.shape != student_attn.shape:
            raise ValueError(
                f"After alignment, teacher shape {teacher_attn.shape} "
                f"!= student shape {student_attn.shape}"
            )
    
        teacher_attn = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + 1e-8)
        student_attn = student_attn / (student_attn.sum(dim=-1, keepdim=True) + 1e-8)


    # Compute loss using appropriate metric for probability distributions
    if loss_type.lower() == 'kl':
        # KL divergence: D_KL(teacher || student)
        # Cast to float32: KL log() on float16 risks underflow for small probabilities
        teacher_attn = teacher_attn.float()
        student_attn = student_attn.float()
        teacher_attn = torch.clamp(teacher_attn, min=1e-6)
        student_attn = torch.clamp(student_attn, min=1e-6)

        # KL divergence expects log-probabilities for student, probabilities for teacher
        kl_reduction = 'batchmean' if reduction == 'mean' else reduction

        loss = F.kl_div(
            student_attn.log(),  # log(student)
            teacher_attn,         # teacher (unnormalized is OK, KL handles it)
            reduction=kl_reduction
        )


    elif loss_type.lower() == 'mse':
        # MSE loss on softmax attention: simpler but doesn't respect probability structure
        loss = F.mse_loss(student_attn, teacher_attn, reduction=reduction)

    elif loss_type.lower() == 'mse_logits':
        # MSE loss on raw logits (before softmax)
        # This captures attention "sharpness" and may be more informative
        if teacher_logits is None or student_logits is None:
            raise ValueError(
                f"loss_type='mse_logits' requires teacher_logits and student_logits, "
                f"but got teacher_logits={teacher_logits is not None}, "
                f"student_logits={student_logits is not None}"
            )

        # Ensure logits are on same device
        if student_logits.device != teacher_logits.device:
            student_logits = student_logits.to(teacher_logits.device)

        # Handle spatial alignment for logits
        if spatial_align and teacher_logits.shape[-1] == 257:
            # Align CLIP logits to FastViT spatial size
            student_n = student_logits.shape[-1]
            target_spatial = int(student_n ** 0.5)
            if target_spatial * target_spatial != student_n:
                raise ValueError(f"Student logits size {student_n} is not a perfect square")

            teacher_logits = align_attention_spatial(teacher_logits, target_size=target_spatial)

        # Handle head dimension mismatch in logits
        if teacher_logits.dim() == 3 and student_logits.dim() == 4:
            # Teacher has no heads, student has heads - average student's heads
            student_logits = student_logits.mean(dim=1)
        elif teacher_logits.dim() == 4 and student_logits.dim() == 3:
            # Teacher has heads, student doesn't - average teacher's heads
            teacher_logits = teacher_logits.mean(dim=1)

        # Ensure dimensions match
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                f"After alignment, teacher_logits shape {teacher_logits.shape} "
                f"!= student_logits shape {student_logits.shape}"
            )

        # Compute MSE loss on logits
        loss = F.mse_loss(student_logits, teacher_logits, reduction=reduction)
    else:
        raise ValueError(
            f"Unsupported loss_type: {loss_type}. "
            f"Use 'kl', 'mse', or 'mse_logits'"
        )

    return loss


# Example usage
if __name__ == '__main__':
    print("Testing attention utilities...")

    # Test 1: Spatial Alignment
    print("\n1. Testing align_attention_spatial:")
    clip_attn = torch.randn(2, 16, 257, 257).softmax(dim=-1)
    aligned = align_attention_spatial(clip_attn, target_size=7)
    print(f"   Input shape: {clip_attn.shape}")
    print(f"   Output shape: {aligned.shape}")
    print(f"   Expected: torch.Size([2, 16, 49, 49])")
    print(f"   Match: {aligned.shape == torch.Size([2, 16, 49, 49])}")

    # Test 2: Distillation Loss
    print("\n2. Testing attention_distillation_loss:")
    teacher = torch.randn(2, 16, 257, 257)
    student = torch.randn(2, 16, 49, 49)
    loss = attention_distillation_loss(teacher, student)
    print(f"   Teacher shape: {teacher.shape}")
    print(f"   Student shape: {student.shape}")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss requires grad: {loss.requires_grad}")

    print("\n✓ All tests passed!")
