#!/bin/bash
#SBATCH --job-name=linearprobe
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6G 
#SBATCH -c8

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
/home/av354855/VLMtoresnet/ml-fastvit-main
# Run unbuffered

python linear_probe.py \
  --model fastvit_sa36 \
  --model-checkpoint checkpoints/CLIPtoResNet/fastvit_sa36_unfrozen/model_best.pth.tar \
  --projector-checkpoint checkpoints/CLIPtoResNet/fastvit_sa36_unfrozen/projector_best.pth.tar \
  --use-projector \
  --dataset aircraft \
  --data-dir /home/c3-0/datasets/fgvc-aircraft-2013b/data \
  --batch-size 128 \
  --workers 4