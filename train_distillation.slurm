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

# Initialize conda in the current shell session (no need for conda init)
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your conda environment
conda activate openmmlab

cd /home/av354855/distillation/VLMtoresnet

# Run your training script
python eva_distillation_imagenet.py