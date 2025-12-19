#!/bin/bash
#SBATCH --job-name=uf_b
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=8

PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

# Load necessary modules
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate fastvit
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory
cd /home/av354855/VLMtoresnet/ml-fastvit-main

# Run unbuffered
python -m torch.distributed.launch --nproc_per_node=2 train_baseline.py \
    /home/c3-0/datasets/ImageNet \
    --model fastvit_sa36_projector \
    --val-set "food101" \
    --validation-data-dir /home/c3-0/datasets/food-101 \
    --validation-eval-interval 2000 \
    --initial-checkpoint Weights/fastvit_sa36.pth.tar \
    --output ./checkpointsunfrozen \
    -b 32 --lr 1e-3 \
    --log-wandb --native-amp --input-size 3 224 224 \
    --drop-path 0.35 --mixup 0 --cutmix 0 \
    --experiment CLIPtoResNet \
    --workers 12 --epochs 50 \

conda deactivate