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

python train_noise.py \
  /home/c3-0/datasets/ImageNet \
  --model fastvit_sa36 \
  --pretrained \
  --epochs 20 \
  --batch-size 128 \
  --noise-experiment \
  --noise-mean-var 0.0033 \
  --noise-step 0.05 \
  --noise-max 1.0 \
  --recovery-acc 80 \
  --noise-val-interval 1000\
  --noise-val-batches 50