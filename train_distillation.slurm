#!/bin/bash
#SBATCH --job-name=distillation_train       # Job name
#SBATCH --output=distillation_%j.out        # Output log file (%j will be replaced with the job ID)
#SBATCH --error=distillation_%j.err         # Error log file
#SBATCH --ntasks=1                          # Number of tasks (1 for single GPU training)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --gres=gpu:1                        # Number of GPUs (adjust as needed)
#SBATCH --mem=16G                           # Memory allocation (adjust as needed)
#SBATCH --time=48:00:00                     # Maximum runtime (adjust as needed)
#SBATCH --partition=gpu                     # Partition name (adjust based on your cluster configuration)

# Load necessary modules (if required)
module load python/3.8
module load cuda

# Activate your Python environment
conda activate openmmlab
# Navigate to the directory containing your script
cd /Users/avinash/Desktop/distillation/VLMtoresnet

# Run your training script
python eva_distillation_imagenet.py