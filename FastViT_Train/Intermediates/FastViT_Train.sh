#!/bin/bash
#SBATCH --job-name=FastViT_2GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 2 tasks (for 2 GPUs)
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=8            # 16 CPUs total
#SBATCH --mem=64G                    # Safe memory amount
#SBATCH --time=48:00:00              # Increased to 48h (2 GPUs is slower than 8)
#SBATCH --output=logs/FastViT_%j.out
#SBATCH --error=logs/FastViT_%j.err

# 1. Optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# 2. Environment
module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate convmix

# 3. Debug Info (Check what we actually got)
echo "Running on $(hostname)"
nvidia-smi

# 4. Run Training
# CHANGED: nproc_per_node=2 matches gres=gpu:2
torchrun --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py /datasets/ImageNet2012nonpub \
    -c config_clip_distill.yaml \
    --output ./output/fastvit_clip_distill \
    --experiment sa36_clip_prompting_2gpu