#!/bin/bash
#SBATCH --job-name=timm-resnet-A2  # Name of your job
#SBATCH --partition=your_gpu_partition  # The partition (queue) you are submitting to
#SBATCH --nodes=1                     # Total number of nodes
#SBATCH --ntasks-per-node=8           # Number of tasks (usually 1 per GPU)
#SBATCH --gres=gpu:8                  # Request 8 GPUs
#SBATCH --cpus-per-task=8             # Number of CPU cores per task (important for dataloaders)
#SBATCH --mem=128G                    # Total memory
#SBATCH --time=48:00:00               # Job run time (hh:mm:ss)
#SBATCH --output=slurm_logs/%x-%j.out # Standard output and error log

# --- 1. Set up your environment ---
# Load any modules you need (e.g., CUDA, Anaconda)
# Example:
# module load cuda/11.8
# module load anaconda/2023.03

# Load necessary modules
module load anaconda3
module load cuda

# ✅ Properly initialize conda in non-interactive mode
eval "$(conda shell.bash hook)"

# Activate your Python environment (e.g., Conda or venv)
source activate openmmlab

# --- 2. Define paths ---
# Set this to the path where your ImageNet 'train' and 'val' folders are
DATA_PATH='/home/c3-0/datasets/ImageNet/train'

# Set this to where you want model checkpoints and logs to be saved
OUTPUT_PATH='/home/av354855/VLMtoresnet'

# --- 3. Run the training command ---
# This is the same command from the previous answer, now run by Slurm.
# We use `srun` to launch the distributed training.
# Slurm's `srun` will automatically handle the distributed setup based on
# the #SBATCH directives above, so we don't need `torch.distributed.launch`.

# NOTE: The timm script automatically detects the Slurm environment.
# You can also use `torch.distributed.launch` if you prefer, but `srun` is cleaner.

echo "Starting TIMM training..."

srun python train.py ${DATA_PATH} \
    --model resnet50 \
    --sched cosine \
    --epochs 300 \
    --lr 5e-3 \
    --batch-size 256 \
    --opt lamb \
    --opt-eps 1e-6 \
    --weight-decay 0.02 \
    --warmup-epochs 5 \
    --warmup-lr 1e-6 \
    --color-jitter 0.4 \
    --aa rand-m7-mstd0.5-inc1 \
    --reprob 0.25 \
    --remode pixel \
    --mixup 0.1 \
    --cutmix 1.0 \
    --loss bce \
    --drop-path 0.05 \
    --ra-reps 4 \
    --amp \
    --output ${OUTPUT_PATH} \
    --log-wandb  # Optional: add this if you use Weights & Biases

echo "Job Finished!"