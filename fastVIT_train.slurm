#!/bin/bash
#SBATCH --job-name=timm_resnet
#SBATCH --output=timm_resnet%j.out
#SBATCH --error=timm_resnet%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# Load necessary modules
module load anaconda3
module load cuda

# ✅ Properly initialize conda in non-interactive mode
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate openmmlab

# Change directory
cd /home/av354855/VLMtoresnet

python -m torch.distributed.launch --nproc_per_node=1 fastvit/train.py \
/home/c3-0/datasets/ImageNet --model fastvit_t8 -b 128 --lr 1e-3 \
--native-amp --mixup 0.2 --output /checkpoints/fastVIT_training \
--input-size 3 256 256

# Optional cleanup
conda deactivate

















