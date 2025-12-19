#!/bin/bash
#SBATCH --job-name=noise
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=8

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

python -m torch.distributed.launch --nproc_per_node=1 train_noise.py \
  /home/c3-0/datasets/ImageNet \
  --model fastvit_sa36 \
  --batch-size 32 \
  --noise-experiment \
  --initial-checkpoint Weights/fastvit_sa36.pth.tar \
  --resume Weights/fastvit_sa36.pth.tar \
  --noise-val-batches 200 \
  --noise-val-interval 2000 \
  --workers 10 \
  --log-wandb