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

# --- FIX STARTS HERE ---

# Source your bash profile to initialize conda
source ~/.bashrc

# Activate your specific conda environment
conda activate openmmlab

# --- FIX ENDS HERE ---

# Change to your working directory
cd /home/av354855/VLMtoresnet

# Run your script (now inside the activated environment)
python -u logit_distillation_cliptoresnet.py

# (Optional but good practice) Deactivate the environment at the end
conda deactivate