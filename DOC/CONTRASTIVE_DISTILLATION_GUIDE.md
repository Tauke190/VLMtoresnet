# Contrastive Distillation Loss Implementation

## Overview

Contrastive distillation combines knowledge distillation with contrastive learning to transfer zero-shot capabilities from CLIP to FastViT. The key idea is to:

1. **Pull together positive pairs**: Maximize similarity between teacher (CLIP) and student (FastViT) features from the SAME image
2. **Push apart negative pairs**: Minimize similarity between teacher and student features from DIFFERENT images
3. **Align final layers**: Use MSE loss to match feature distributions
4. **Transfer CLIP knowledge**: Use CLIP text alignment loss to maintain zero-shot capability

---

## Components Implemented

### 1. **ContrastiveDistillationLoss** (`Functions/contrastive_distillation_loss.py`)

Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss - the core of SimCLR.

#### How NT-Xent Works:

```
Given:
- Student features: [B, D]
- Teacher features: [B, D]

Step 1: Normalize both feature sets
student_norm = normalize(student_features)  # [B, D]
teacher_norm = normalize(teacher_features)  # [B, D]

Step 2: Compute similarity matrix
sim_matrix = student_norm @ teacher_norm.T / temperature  # [B, B]
    where sim_matrix[i,j] = dot_product(student_i, teacher_j)

Step 3: Create positive pair labels
labels = [0, 1, 2, ..., B-1]  # Diagonal elements are positive pairs

Step 4: Cross-entropy loss
loss = cross_entropy(sim_matrix, labels)
    This maximizes sim_matrix[i,i] (positive pairs)
    while minimizing sim_matrix[i,j] for i≠j (negative pairs)
```

#### Why This Works for Distillation:

- **Contrastive learning** forces the student to learn a feature space similar to the teacher
- **Temperature scaling** (default: 0.07) controls sharpness of similarity distribution
  - **Lower temp** (< 0.07): Sharper, more discriminative
  - **Higher temp** (> 0.07): Softer, more gradual
- **Symmetric version** available: treats both directions equally for stability

#### Code Example:

```python
from Functions.contrastive_distillation_loss import ContrastiveDistillationLoss

loss_fn = ContrastiveDistillationLoss(temperature=0.07)

# In training loop:
student_features = model(batch_images)          # [B, D]
teacher_features = clip_model(batch_images)     # [B, D]

contrastive_loss = loss_fn(student_features, teacher_features)
total_loss = base_loss + contrastive_loss
```

---

### 2. **Loss Manager Integration** (`Functions/losses.py`)

The `create_contrastive_distillation_loss_manager()` function creates a composable loss that combines:

```
Total Loss = Base Loss + CLIP Loss + MSE Loss + Contrastive Loss

Where:
- Base Loss: Classification loss (CrossEntropy)
- CLIP Loss: Text alignment for zero-shot capability
- MSE Loss: Final layer feature alignment  
- Contrastive Loss: NT-Xent loss for positive pair pulling
```

#### LossManager Architecture:

```
LossManager (base_loss_fn)
    ├── Base Loss (always computed)
    ├── CLIP Loss (if text features available)
    ├── MSE Loss (feature distribution matching)
    └── Contrastive Distill Loss (new!)
```

---

### 3. **Training Setup** (`train_baseline.py`)

Updated to pass contrastive parameters:

```python
loss_manager = get_loss_manager_for_method(
    method="contrastive_distillation",
    base_loss_fn=train_loss_fn,
    clip_loss_fn=clip_loss_fn,
    clip_text_features=clip_text_features,
    clip_logit_scale=clip_logit_scale,
    contrastive_weight=args.contrastive_weight,          # NEW
    contrastive_temperature=args.contrastive_temperature, # NEW
)
```

---

### 4. **Command-Line Arguments** (`Functions/argument.py`)

Added three new parameters:

```bash
--method contrastive_distillation
--contrastive-weight 1.0              # Scale of contrastive loss
--contrastive-temperature 0.07        # NT-Xent temperature
```

---

## Step-by-Step Explanation

### Step 1: Feature Extraction
```python
# Forward pass through models
batch_images = torch.randn(32, 3, 224, 224)
student_output, student_logits, student_features = student_model(batch_images)
        # student_features: [B, D] projected features

with torch.no_grad():
    teacher_output = teacher_model(batch_images)
    teacher_features = teacher_output  # CLIP image features [B, 512 or 768]
```

### Step 2: Loss Computation
```python
# The LossManager computes all losses
total_loss, loss_dict = loss_manager.compute(
    output=student_logits,                    # [B, num_classes]
    target=batch_labels,                      # [B]
    projected_embed=student_features,         # [B, D]
    clip_image_features=teacher_features,     # [B, 512/768]
)

print(loss_dict)
# Output:
# {
#     'Base Loss': 2.3,           # CrossEntropy
#     'CLIP Loss': 0.15,          # Text alignment
#     'MSE Loss': 0.08,           # Feature alignment
#     'Contrastive Distill Loss': 0.42   # NEW!
# }
```

### Step 3: Backward Pass
```python
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

# Gradients flow through:
# - student_logits (base loss)
# - student_features (MSE + Contrastive)
# - student_model parameters
```

---

## Usage Examples

### Example 1: Basic Training

```bash
cd /home/av354855/projects/VLMtoresnet

IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
VAL_PATH=/mnt/SSD2/food-101
OUTPUT=./checkpoints
NUM_GPU=2

CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH \
    --model fastvit_sa36_adapter \
    --val-set food101 \
    --validation-data-dir $VAL_PATH \
    --output $OUTPUT \
    --freeze-backbone \
    --method contrastive_distillation \
    --contrastive-weight 1.0 \
    --contrastive-temperature 0.07 \
    -b 32 --lr 1e-3 --epochs 50
```

### Example 2: Adjusting Contrastive Loss Strength
```bash
# Strong contrastive signal (2x weight)
--contrastive-weight 2.0 \
--contrastive-temperature 0.07

# Soft contrastive signal (0.5x weight, higher temp)
--contrastive-weight 0.5 \
--contrastive-temperature 0.1
```

### Example 3: Comparing with Distillation
```bash
# Standard distillation (MSE only)
--method distillation

# Contrastive distillation (MSE + NT-Xent)
--method contrastive_distillation --contrastive-weight 1.0
```

---

## How It Improves Training

### Problem with Standard Distillation (MSE Loss Only):

```
Student features might have same distribution as teacher
but in a rotated/scaled subspace.
This doesn't guarantee good alignment in the direction 
that matters for zero-shot transfer.
```

### Solution with Contrastive Distillation:

```
1. MSE Loss pulls global distributions together
2. Contrastive Loss pulls POSITIVE PAIRS (same image) together
3. Contrastive Loss pushes NEGATIVE PAIRS apart
4. Result: Student learns a feature space where:
   - Same images have similar teacher-student features
   - Different images have dissimilar features
   - This preserves CLIP's semantic structure
```

### Expected Benefits:

1. **Better Zero-Shot Transfer**: Maintains CLIP's semantic alignment
2. **Faster Convergence**: Contrastive learning is efficient
3. **More Stable**: Multiple loss signals help regularization
4. **Preserves Class Relationships**: Pulling positive pairs enforces class structure

---

## Mathematical Details

### NT-Xent Loss Formula:

```
L = -log( exp(sim(student_i, teacher_i) / τ) / Σ_j exp(sim(student_i, teacher_j) / τ) )

Where:
- sim(a, b) = (a · b) / (||a|| * ||b||)  [cosine similarity]
- τ [tau] = temperature parameter
- i = positive pair index
- j = all indices (includes negatives and positive)
```

### In Probability Terms:

```
Positive pair probability = softmax(similarities / τ)[positive_idx]

The loss minimizes the negative log of this probability.
Higher positive similarity → lower loss
Lower negative similarities → lower loss
```

### Temperature Effect:

```
Temperature = 0.01:  Very sharp distribution
         ↓
     Steep gradients ↓
     Harder negatives focused
 
Temperature = 0.07:  Standard (recommended)
         ↓
     Balanced gradients
     
Temperature = 0.3:   Softer distribution
         ↓
     Gentle gradients
     Easier training, less discriminative
```

---

## Troubleshooting

### Issue 1: Loss becomes NaN

**Causes**: 
- Features not properly normalized
- Temperature too low with large batch size
- Mixing zero-norm features

**Solution**:
```python
# Check feature norms
print(f"Student norm: {student_features.norm(dim=1).mean()}")
print(f"Teacher norm: {teacher_features.norm(dim=1).mean()}")

# If not ~1.0, there's an issue in feature extraction
```

### Issue 2: Contrastive Loss Dominates

**Causes**:
- Contrastive weight too high
- Temperature too low

**Solution**:
```bash
# Reduce weight
--contrastive-weight 0.5

# Increase temperature
--contrastive-temperature 0.1
```

### Issue 3: Zero-Shot Accuracy Doesn't Improve

**Causes**:
- CLIP loss weight might be too low
- Contrastive loss overrides CLIP alignment

**Solution**:
```bash
# Balance the losses
--contrastive-weight 1.0  # Not too aggressive
--method contrastive_distillation  # Includes CLIP loss
```

---

## Files Modified

1. **Created**: `Functions/contrastive_distillation_loss.py`
   - `ContrastiveDistillationLoss` class
   - `ContrastiveDistillationLossSymmetric` class

2. **Modified**: `Functions/losses.py`
   - Added `create_contrastive_distill_loss()`
   - Added `create_contrastive_distillation_loss_manager()`
   - Updated `get_loss_manager_for_method()`

3. **Modified**: `Functions/argument.py`
   - Added `--method` choice: `contrastive_distillation`
   - Added `--contrastive-weight` argument
   - Added `--contrastive-temperature` argument

4. **Modified**: `train_baseline.py`
   - Pass contrastive parameters to loss manager

---

## Next Steps

### Experimentation:

```bash
# Benchmark contrastive vs standard distillation
--method distillation  # Baseline
--method contrastive_distillation --contrastive-weight 1.0  # New

# Test different temperatures
--contrastive-temperature 0.05  # Sharper
--contrastive-temperature 0.1   # Softer
```

### Monitoring:

```python
# In training output, watch for:
'Base Loss': decreasing
'CLIP Loss': stable or slightly decreasing  
'MSE Loss': decreasing
'Contrastive Distill Loss': decreasing  ← Key indicator
```

### Validation:

```python
# Test zero-shot accuracy:
zeroshot_accuracy = run_zeroshot_eval(model, clip_model, test_loader)
# Should be comparable to or better than standard distillation
```
