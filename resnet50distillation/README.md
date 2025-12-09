# Distilling CLIP Zero-Shot Capability into ResNet

## Summary
Exploring how to transfer zero-shot image classification from a large CLIP vision–language model into a compact ResNet backbone for faster, lighter deployment.

## Why
- Smaller model, lower latency, easier edge / batch deployment.
- Keeps prompt-based zero-shot classification without full VLM overhead.

## Distillation Strategies (Under Exploration)
- Final feature regression (student matches teacher image embedding).
- Intermediate layer alignment (multi-stage feature guidance).
- Contrastive + logit distillation (class/text-aware similarity shaping).
- Masked / partial feature prediction (robustness and regularization).

## Use
1. Run chosen distillation script.
2. Cache text embeddings once.
3. Perform zero-shot classification with distilled student.

## Status
Active experiments comparing accuracy vs cost across strategies.

# VLMtoresnet

This repository contains scripts for various knowledge distillation techniques from CLIP to ResNet, including contrastive, final feature, intermediate feature, and masked generative distillation.

## Setup

1. **Clone the repository** and install dependencies:
    ```bash
    git clone <this-repo-url>
    cd VLMtoresnet
    pip install -r requirements.txt
    ```

2. **Prepare datasets**  
   Place your ImageNet and Oxford-IIIT Pet datasets in the following structure (or provide custom paths):
   ```
   ~/data/datasets/imagenet/train/
   ~/data/datasets/imagenet/validation/
   ~/data/datasets/oxford_pet/val/
   ```

## Running Distillation Scripts

Use the main launcher script `KD_training.py` to run any distillation method:

```bash
python KD_training.py --distillation <type> [--train_dir <train_path>] [--val_dir <val_path>]
```

- `<type>`: One of `contrastive`, `finalfeature`, `intermediate`, `masked_generative`
- `--train_dir` and `--val_dir` are optional. If not provided, defaults from the scripts will be used.

**Example:**
```bash
python KD_training.py --distillation finalfeature
```
or with custom dataset paths:
```bash
python KD_training.py --distillation contrastive --train_dir /path/to/train --val_dir /path/to/val
```

## Evaluating Distilled Models

To evaluate a distilled model checkpoint, use:
```bash
python KD_evaluate.py --checkpoint <path_to_checkpoint> --dataset <imagenet|oxfordpet>
```

**Example:**
```bash
python KD_evaluate.py --checkpoint distilled_checkpoints/finalfeature_distillation_cliptoresnet.pt --dataset imagenet
```

## Download Pretrained Distilled Models

You can download distilled model checkpoints from the following link:

[Download distilled models](https://drive.google.com/drive/folders/1DEJsbp_SOT6L-m9Hu-JrxyUsZoHnDUpY)

---

For any issues, please open an issue on this repository.