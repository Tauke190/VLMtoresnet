#!/bin/bash
#SBATCH --job-name=logit_distillation_train
#SBATCH --output=logit_distillation_%j.out
#SBATCH --error=logit_distillation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00

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
python -u evals/linearprobeeval.py --checkpoint ../checkpoint checkpoint_epoch_4.pth imagenet

# Optional cleanup
conda deactivate
