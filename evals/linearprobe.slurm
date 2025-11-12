#!/bin/bash
#SBATCH --job-name=final_feaure_linearprobe
#SBATCH --output=final_feaure_linearprobe%j.out
#SBATCH --error=final_feaure_linearprobe%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
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
python -u linearprobeeval.py --checkpoint ../distilled_checkpoints/finalfeature_distillation_cliptoresnet.pt --dataset imagenet


# Optional cleanup
conda deactivate