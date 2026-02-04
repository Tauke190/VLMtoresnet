# Copilot Instructions for VLMtoresnet

## Project Overview
This repository focuses on training, evaluating, and distilling vision-language models (VLMs) using FastViT and ResNet architectures, with CLIP-based zero-shot and linear probing workflows. The goal is to enable efficient, prompt-based classification on edge devices by distilling CLIP capabilities into compact ResNet backbones.

## Key Components & Structure
- **Top-level scripts**: Training (`train_baseline.py`, `train.py`, `train_noise.py`), validation (`validate.py`), zero-shot evaluation (`zeroshot_eval.py`), model export (`export_model.py`), and linear probing (`linear_probe.py`, `linear_probing_demo.py`).
- **CLIP/**: Contains CLIP integration, backbone selection, and zero-shot evaluation logic. See `CLIP/README.md` for supported datasets and backbones.
- **Functions/**: Shared utilities for argument parsing, training, evaluation, and loss functions.
- **models/**: FastViT and ResNet model definitions and modules.
- **resnet50distillation/**: Distillation strategies, scripts, and documentation for transferring CLIP zero-shot capabilities to ResNet. See `resnet50distillation/README.md` for details.
- **configs/**: Configuration files for different datasets and experiments.
- **scripts/**: Slurm and shell scripts for cluster/distributed training and evaluation.

## Developer Workflows
- **Environment**: Use the `fastvit` Conda environment. Activate with `conda activate /home/av354855/miniconda3/envs/fastvit`.
- **Install dependencies**: `pip install -r requirements.txt` (root and some submodules).
- **CLIP local setup**: In `CLIP/`, run `pip install -e .` for editable install.
- **Dataset preparation**: Place datasets under `~/data/datasets/imagenet/train/` or provide custom paths.
- **Training**: Use `train_baseline.py` for reproducible FastViT experiments, or `train.py` for original FastViT. For noisy data, use `train_noise.py`.
- **Distillation**: Run scripts in `resnet50distillation/` for CLIP-to-ResNet knowledge transfer. Cache text embeddings before zero-shot classification.
- **Zero-shot evaluation**: Run `python CLIP/resolution_zero_shot.py --dataset [name] --image_resolution [16|32|64|128|224] --batch_size [N] --backbone [clip_backbone]`.
- **Linear probing**: Use `linear_probe.py` or `linear_probing_demo.py` for feature evaluation.
- **Cluster jobs**: Submit Slurm scripts in `scripts/` for distributed training/evaluation.

## Patterns & Conventions
- **Backbone selection**: Use CLI arguments to specify model backbone (e.g., ViT-B/16, RN50).
- **Supported datasets**: See `CLIP/README.md` for full list; dataset names must be lowercase.
- **Image resolution**: Accepts 16, 32, 64, 128, 224 (default: 224).
- **Distillation strategies**: Final feature regression, intermediate alignment, contrastive/logit distillation, masked prediction (see `resnet50distillation/README.md`).
- **Logging**: Custom logger in `CLIP/logger.py`.
- **Config files**: Use YAML configs in `configs/` for experiment reproducibility.

## Integration Points
- **External dependencies**: CLIP, FastViT, PyTorch, CoreML (for export), Slurm (for cluster jobs).
- **Cross-component communication**: Models and utilities are modular; scripts import from `Functions/`, `models/`, and `CLIP/` as needed.

## Example: Zero-Shot Evaluation
```bash
python CLIP/resolution_zero_shot.py \
  --dataset imagenet1k \
  --image_resolution 224 \
  --batch_size 64 \
  --backbone ViT-B/16
```

## Example: Distillation Workflow
1. Run distillation script in `resnet50distillation/`.
2. Cache text embeddings.
3. Evaluate distilled student with zero-shot classification.

---
For unclear or incomplete sections, please provide feedback or specify which workflows or conventions need further documentation.
