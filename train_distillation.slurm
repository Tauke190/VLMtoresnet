#!/bin/bash
#SBATCH --job-name=distillation_train
#SBATCH --output=distillation_%j.out
#SBATCH --error=distillation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable

# Load necessary modules
module load anaconda
module load python
module load cuda

# --- FIX STARTS HERE ---

# Source your bash profile to initialize conda
source ~/.bashrc

# Activate your specific conda environment
conda activate openmmlab

# --- FIX ENDS HERE ---

# Change to your working directory
cd /home/av354855/distillation/VLMtoresnet

# Run your script (now inside the activated environment)
python eva_distillation_imagenet.py

# (Optional but good practice) Deactivate the environment at the end
conda deactivate