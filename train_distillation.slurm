#!/bin/bash
#SBATCH --job-name=distillation_train
#SBATCH --output=distillation_%j.out
#SBATCH --error=distillation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0:00:15
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable

module load anaconda
module load python
module load cuda

# Activate the conda environment using full path (avoids conda.sh issues)
# Replace with your actual conda base path if different
/home/av354855/.conda/bin/conda activate openmmlab

# Navigate to your project directory
cd /home/av354855/distillation/VLMtoresnet

# Run your training script
python eva_distillation_imagenet.py
