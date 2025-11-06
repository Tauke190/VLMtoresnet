#!/bin/bash
#SBATCH --job-name=logit_distillation_train
#SBATCH --output=logit_distillation_%j.out
#SBATCH --error=logit_distillation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --constraint=ampere        # ✅ request Ampere GPUs only

# Load necessary modules
module load anaconda3
module load cuda

# ✅ Properly initialize conda in non-interactive mode
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate openmmlab

# Change directory
cd /home/av354855/VLMtoresnet

# Run unbuffered
python -u timm_resnet50.py

# 🧠 Log GPU info before training
echo "===== GPU Information ====="
nvidia-smi
echo "==========================="

# Optional cleanup
conda deactivate
