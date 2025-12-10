#!/bin/bash
#SBATCH --job-name=fastvit_unfrozen
#SBATCH --output=fastvit%j.out
#SBATCH --error=fastvit%j.err
#SBATCH --cpus-per-task=2
#SBATCH --constraint=gmem48
#SBATCH --gres=gpu:ampere:2
#SBATCH --mem=48G

# Load necessary modules
module load anaconda3
module load cuda

# ? Properly initialize conda in non-interactive mode
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate fastvit
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# export CUDA_VISIBLE_DEVICES=2,3
echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory
cd /home/av354855/fastvit

# Run unbuffered

# python -u train_baseline.py \
#     /home/c3-0/datasets/ImageNet \
#     --aircraft-data-dir /home/c3-0/datasets/fgvc-aircraft-2013b/data \
#     --model fastvit_sa36 \
#     --initial-checkpoint fastvit_sa36.pth.tar \
#     -b 128 \
#     --lr 1e-3 \
#     --log-wandb \
#     --native-amp \
#     --output ./checkpoints \
#     --input-size 3 256 256 \
#     --mixup 0 --cutmix 0 \
#     --drop-path 0.35 \
#     --aircraft-eval-interval 1000 \
#     --experiment CLIPtoResNet \
#     --workers 4 \
#     --epochs 50

python -m torch.distributed.launch --nproc_per_node=2 train_baseline.py \
    /home/c3-0/datasets/ImageNet \
    --model fastvit_sa36 \
    --aircraft-data-dir /home/c3-0/datasets/fgvc-aircraft-2013b/data \
    --initial-checkpoint fastvit_sa36.pth.tar \
    -b 128 \
    --lr 1e-3 \
    --log-wandb \
    --native-amp \
    --output ./checkpoints \
    --input-size 3 256 256 \
    --drop-path 0.35 \
    --mixup 0 --cutmix 0 \
    --experiment CLIPtoResNet \
    --workers 4 \
    --epochs 50 \
    --aircraft-eval-interval 1000 \

# Optional cleanup
conda deactivate
