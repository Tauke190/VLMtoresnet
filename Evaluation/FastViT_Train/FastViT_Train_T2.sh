#!/bin/bash
#SBATCH --job-name=T2_FastViT_CLIP_KD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/T2_FastViT_CLIP_KD_%j.out
#SBATCH --error=logs/T2_FastViT_CLIP_KD_%j.err

mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate convmix

echo "Running on $(hostname)"
nvidia-smi

torchrun --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    Fastvit_Train2.py
