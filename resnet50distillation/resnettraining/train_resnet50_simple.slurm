#!/bin/bash
#SBATCH --job-name=timm_resnet
#SBATCH --output=timm_resnet%j.out
#SBATCH --error=timm_resnet%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=100:00:00

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
python -u train_resnet50_advanced.py

# Optional cleanup
conda deactivate
