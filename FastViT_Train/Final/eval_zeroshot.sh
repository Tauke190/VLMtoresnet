#!/bin/bash
#SBATCH --job-name=ZeroShot_FastViT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ZeroShot_FastViT_%j.out
#SBATCH --error=logs/ZeroShot_FastViT_%j.err

set -euo pipefail

# Create logs directory
mkdir -p logs

# Load environment
module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate convmix

# Print info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="
echo ""

# Run evaluation
python eval_zeroshot.py

echo ""
echo "Evaluation completed!"
