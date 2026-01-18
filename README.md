# VLMtoresnet

This repository contains scripts and tools for training, evaluating, and exporting vision-language models using FastViT and ResNet architectures, with CLIP-based methods and linear probing.



## Main Files

- **export_model.py**: Export FastViT models to CoreML for mobile inference.
- **linear_probe.py**: Linear probing with logistic regression to evaluate backbone features.
- **linear_probing_demo.py**: Demonstration of linear probing workflow and feature extraction.
- **train_baseline.py**: fastvit Baseline ImageNet training script for reproducible experiments.
- **train_noise.py**: Training with noisy data/labels to estimate time for training.
- **train.py**: Original FastVIT training script.
- **validate.py**: Validate models or checkpoints on ImageNet-like datasets using logits.
- **zeroshot_eval.py**: Zero-shot evaluation using CLIP-style text features and templates.
- **Scripts/train.slurm**: Slurm batch script for distributed training of the model and its variant on a cluster environment.

## Evaluation Scripts

This repository provides evaluation scripts at Scripts/eval.sh for

- **Zero-shot evaluation**: Evaluate models using CLIP-style text features and templates without fine-tuning.
- **Linear probe**: Assess backbone features using logistic regression or similar classifiers.
- **Logit-based classification**: Perform classification directly using model logits for analysis and benchmarking.
