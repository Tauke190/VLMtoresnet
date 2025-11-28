#!/bin/bash
#SBATCH --job-name=ConvMixer_Eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      
#SBATCH --gres=gpu:1                 
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                   
#SBATCH --time=12:00:00             
#SBATCH --output=logs/ConvMixer_%j.out
#SBATCH --error=logs/ConvMixer_%j.err

# Environment optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p logs

# Load environment
module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate convmix

# Debug info
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi

# Run evaluation
python ConvMixer.py

echo "Evaluation completed!"