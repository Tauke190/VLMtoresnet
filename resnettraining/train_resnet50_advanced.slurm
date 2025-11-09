#!/bin/bash
#SBATCH --job-name=timm_resnet50
#SBATCH --output=timm_resnet50%j.out
#SBATCH --error=timm_resnet50%j.err
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
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
python train_resnet50_advanced.py

# 🧠 Log GPU info before training
echo "===== GPU Information ====="
nvidia-smi
echo "==========================="

# Optional cleanup
conda deactivate
