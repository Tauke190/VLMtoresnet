"""
Contrastive Distillation Loss for CLIP-to-FastViT Knowledge Transfer.

This module implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
for pulling together positive pairs (teacher-student from same image) while pushing 
apart negative pairs (teacher-student from different images).

The main idea:
1. Normalize teacher (CLIP) and student (FastViT) image features
2. Compute similarity matrix between all teacher-student pairs
3. Use NT-Xent loss to maximize similarity for positive pairs, minimize for negatives
4. Apply temperature scaling for controllable smoothness

Reference:
- SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020)
- CLIP: Learning Transferable Models for Code (Radford et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDistillationLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning between teacher and student features.
    
    Pulls teacher-student pairs from the same image together while pushing apart
    pairs from different images.
    
    Usage:
        loss_fn = ContrastiveDistillationLoss(temperature=0.07)
        teacher_features = [B, D]  # CLIP image features
        student_features = [B, D]  # FastViT projected features
        loss = loss_fn(student_features, teacher_features)
    """
    
    def __init__(self, temperature=0.07, reduction='mean'):
        """
        Initialize contrastive distillation loss.
        
        Args:
            temperature: Temperature parameter for scaling logits. 
                        Lower values make the loss sharper. Default: 0.07 (from SimCLR)
            reduction: How to reduce the loss. Options: 'mean', 'sum'. Default: 'mean'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, student_features, teacher_features):
        """
        Compute NT-Xent contrastive loss.
        
        Args:
            student_features: Student model features [B, D]
            teacher_features: Teacher (CLIP) model features [B, D]
            
        Returns:
            loss: Scalar tensor containing contrastive loss
        """
        # Normalize features
        student_features = F.normalize(student_features, dim=1, p=2)  # [B, D]
        teacher_features = F.normalize(teacher_features, dim=1, p=2)  # [B, D]
        
        batch_size = student_features.shape[0]
        
        # Compute similarity matrix: [B, B]
        # logits_ij = student_i · teacher_j
        logits = torch.matmul(student_features, teacher_features.t()) / self.temperature
        
        # Create labels for positive pairs
        # Positive pair: diagonal elements (i, i)
        labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)
        
        # Cross-entropy loss: minimize log-softmax of positive pairs
        # log_softmax normalizes over all pairs, encouraging high similarity for positives
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss


class ContrastiveDistillationLossSymmetric(nn.Module):
    """
    Symmetric NT-Xent loss that treats both directions equally.
    
    Computes both:
    - Cross-entropy from student perspective (student -> teacher)
    - Cross-entropy from teacher perspective (teacher -> student)
    
    This is more stable and symmetric than one-way contrastive loss.
    
    Usage:
        loss_fn = ContrastiveDistillationLossSymmetric(temperature=0.07)
        loss = loss_fn(student_features, teacher_features)
    """
    
    def __init__(self, temperature=0.07, reduction='mean'):
        """
        Initialize symmetric contrastive distillation loss.
        
        Args:
            temperature: Temperature parameter for scaling logits. Default: 0.07
            reduction: How to reduce the loss. Default: 'mean'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, student_features, teacher_features):
        """
        Compute symmetric NT-Xent contrastive loss.
        
        Args:
            student_features: Student model features [B, D]
            teacher_features: Teacher (CLIP) model features [B, D]
            
        Returns:
            loss: Scalar tensor containing symmetric contrastive loss
        """
        # Normalize features
        student_features = F.normalize(student_features, dim=1, p=2)  # [B, D]
        teacher_features = F.normalize(teacher_features, dim=1, p=2)  # [B, D]
        
        batch_size = student_features.shape[0]
        
        # Compute similarity matrices
        # logits_s2t[i, j] = student_i · teacher_j
        logits_s2t = torch.matmul(student_features, teacher_features.t()) / self.temperature  # [B, B]
        # logits_t2s[i, j] = teacher_i · student_j
        logits_t2s = torch.matmul(teacher_features, student_features.t()) / self.temperature  # [B, B]
        
        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size, device=logits_s2t.device, dtype=torch.long)
        
        # Compute loss from both directions and average
        loss_s2t = F.cross_entropy(logits_s2t, labels, reduction=self.reduction)
        loss_t2s = F.cross_entropy(logits_t2s, labels, reduction=self.reduction)
        
        loss = (loss_s2t + loss_t2s) / 2.0
        
        return loss
