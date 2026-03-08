# Contrastive Distillation - Quick Reference

## What It Does

```
Standard Distillation:
Student Features ──MSE──→ Teacher Features
(pixel-space alignment)

Contrastive Distillation:
Student Features ──MSE──→ Teacher Features
                  ↓
              NT-Xent Loss
              (semantic space alignment)
              ↓
        - Pull positive pairs (same image)
        - Push negative pairs (diff image)
        - Preserve CLIP's structure
```

## Visual: How NT-Xent Loss Works

```
Batch of 4 images with their features:

┌─────────────────────────────────────────────────┐
│ Image 0: BIRD                                   │
│ Student0 ─┐                                     │
│           ├──→ Similarity Matrix [4×4]          │
│ Teacher0 ─┘                                     │
│                                                 │
│       Student→Teacher dot products              │
│       ┌─────┬─────┬─────┬─────┐               │
│       │ 8.5 │ 1.2 │ 0.8 │ 1.1 │  ← Bird      │
│       ├─────┼─────┼─────┼─────┤               │
│       │ 1.0 │ 7.8 │ 1.5 │ 0.9 │  ← Car       │
│       ├─────┼─────┼─────┼─────┤               │
│       │ 0.7 │ 1.3 │ 7.2 │ 1.0 │  ← Dog       │
│       ├─────┼─────┼─────┼─────┤               │
│       │ 1.2 │ 0.8 │ 1.1 │ 8.1 │  ← Cat       │
│       └─────┴─────┴─────┴─────┘               │
│        Bird  Car   Dog   Cat                   │
│     (positive pairs on diagonal!)              │
│                                                 │
│ Loss = -log( softmax(similarities)[diagonal] ) │
│                                                 │
│ ✓ Increases diagonal similarities (positive)   │
│ ✓ Decreases off-diagonal (negative)            │
└─────────────────────────────────────────────────┘
```

## Temperature Parameter

```
Temperature τ controls how "sharp" the loss is:

τ = 0.01   │  Very Sharp    → Steep gradients, hard training
           │                → Focuses on hardest negatives
           ├─ τ = 0.07 ─────→ Default/Recommended
           │                → Balanced
           │  
           └─ τ = 0.3  ──→  Soft              → Gentle gradients
                            → Easier training
```

## Command-Line Usage

### Basic (Recommended)
```bash
--method contrastive_distillation
```

### Custom Weights
```bash
# Make contrastive loss stronger
--method contrastive_distillation \
--contrastive-weight 2.0

# Make contrastive loss weaker  
--method contrastive_distillation \
--contrastive-weight 0.5
```

### Fine-Tune Temperature
```bash
# Sharper loss (more discriminative)
--contrastive-temperature 0.05

# Standard (recommended)
--contrastive-temperature 0.07

# Softer loss (easier training)
--contrastive-temperature 0.1
```

## Full Example

```bash
cd /home/av354855/projects/VLMtoresnet

CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch \
    --nproc_per_node=2 train_baseline.py \
    /mnt/SSD2/ImageNet1k/ \
    --model fastvit_sa36_adapter \
    --method contrastive_distillation \
    --contrastive-weight 1.0 \
    --contrastive-temperature 0.07 \
    --val-set food101 \
    --validation-data-dir /mnt/SSD2/food-101 \
    --freeze-backbone \
    -b 32 --lr 1e-3 --epochs 50
```

## Loss Components Breakdown

```
Total Loss = L_base + L_clip + L_mse + L_contrastive

L_base       = CrossEntropy Loss
               Trains classification head
               Always present

L_clip       = CLIP Text Alignment Loss  
               Maintains zero-shot capability
               Uses pre-computed CLIP text features

L_mse        = MSE(Student Features, Teacher Features)
               Aligns feature distributions
               MSE-based distillation

L_contrastive = NT-Xent(Student Feats, Teacher Feats)
                NEW!
                - Pulls same-image pairs together
                - Pushes different-image pairs apart
                - Preserves semantic structure
```

## Debug: Check Loss Values

```python
# During training, loss_dict contains:
{
    'Base Loss': 2.34,                      # Should decay
    'CLIP Loss': 0.12,                      # Should be stable
    'MSE Loss': 0.08,                       # Should decay
    'Contrastive Distill Loss': 0.41        # Should decay ✓
}

# Good signs:
# ✓ All losses decreasing over time
# ✓ Contrastive loss ~0.5-1.0 range
# ✓ Base loss dominant initially, then balanced

# Bad signs:
# ✗ Contrastive loss = NaN
# ✗ Contrastive loss >> other losses
# ✗ Contrastive loss stuck at high value
```

## When to Use Contrastive Distillation

```
✓ Use when:
  - Transferring from CLIP (semantic learning)
  - Training with frozen backbone (limited updates)
  - Need to preserve zero-shot capability
  - Have enough GPU memory for larger batches
  
✗ Skip when:
  - Training from scratch (regular distillation better)
  - Working with small batch sizes (<16) 
  - Very limited computational resources
```

## Key Differences from Other Methods

```
Method               │ Losses Used
─────────────────────┼──────────────────────────────
default              │ Base Loss Only
baseline             │ Base + CLIP Text
distillation         │ Base + CLIP Text + MSE
attention_distill    │ Base + CLIP Text + Attn
contrastive_distill  │ Base + CLIP Text + MSE + Contrastive ✓
                     │ (Recommended for CLIP transfer)
```

## What to Monitor

```
Training Progress:

Epoch 1:   Base Loss: 4.5  CLIP: 0.20  MSE: 0.50  Contrastive: 3.2
Epoch 10:  Base Loss: 2.1  CLIP: 0.15  MSE: 0.25  Contrastive: 1.8
Epoch 30:  Base Loss: 0.8  CLIP: 0.12  MSE: 0.08  Contrastive: 0.4
Epoch 50:  Base Loss: 0.3  CLIP: 0.11  MSE: 0.04  Contrastive: 0.2
                            ↑ All decreasing = Good!

Zero-Shot Eval:
- Track zero-shot accuracy on validation set
- Should improve over baseline distillation
- Plateau indicates convergence
```
