#!/bin/bash
#SBATCH --job-name=masked
#SBATCH --output=masked%j.out
#SBATCH --error=masked%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:12
#SBATCH --mem=8G
#SBATCH --time=72:00:00

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
python -u masked_generative_distillation.py

# Optional cleanup
conda deactivate