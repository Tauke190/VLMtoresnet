#!/bin/bash
#SBATCH --job-name=fastvit
#SBATCH --output=fastvit%j.out
#SBATCH --error=fastvit%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=gmem48
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48G
#SBATCH --time=100:00:00

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

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory
cd /home/av354855/fastvit

# Run unbuffered
# python -m train /home/c3-0/datasets/ImageNet --resume fastvit_ma36.pth.tar --no-resume-opt --model fastvit_ma36 -b 128 --lr 1e-3 --native-amp --mixup 0.2 --output ./checkpoints --input-size 3 256 256 --log-wandb --experiment CLIPtoResNet --workers 4 --epochs 50 --val-interval 1000


python -m torch.distributed.launch --nproc_per_node=2 train.py \
/home/c3-0/datasets/ImageNet --model fastvit_ma36 -b 128 --lr 1e-3 \
--native-amp --output ./checkpoints \
--input-size 3 256 256 --drop-path 0.35


# Optional cleanup
conda deactivate
