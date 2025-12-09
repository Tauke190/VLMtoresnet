#!/bin/bash
#SBATCH --job-name=Final_FastViT_1node_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --gres=gpu:2                 
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/Final_1node_%j.out
#SBATCH --error=logs/Final_1node_%j.err

# 1. Runtime/env settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# 2. Conda env
module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate convmix

# 3. Debug info + periodic nvidia-smi logging
echo "Running on $(hostname)"
echo "Starting periodic nvidia-smi logging..."
(
    while true; do
        echo "===== nvidia-smi @ $(date) ====="
        nvidia-smi
        sleep 300   # log every 5 minutes
    done
) &

# 4. Launch torchrun on a single node with 2 GPUs
torchrun \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py /datasets/ImageNet2012nonpub \
    -c ../config_clip_distill.yaml \
    --output ./output/fastvit_clip_distill \
    --experiment sa36_clip_prompting_1node_2gpu