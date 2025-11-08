#!/bin/bash
#SBATCH --job-name=final_feature
#SBATCH --output=final_feature%j.out
#SBATCH --error=final_feature%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=48:00:00

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
python -u logit_distillation_cliptoresnet.py

# Optional cleanup
conda deactivate