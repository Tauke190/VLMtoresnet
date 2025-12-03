#!/bin/bash
#SBATCH --job-name=LinearProbe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/LinearProbe_%j.out
#SBATCH --error=logs/LinearProbe_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

module load anaconda/anaconda-2024.10
eval "$(conda shell.bash hook)"
conda activate convmix

# Create logs directory if it doesn't exist
mkdir -p logs

# Define models to evaluate
MODELS=("clip_vitl14" "eva02_clip" "fastvit_sa36" "convmixer_768_32" "scalekd_resnet50")

# Run evaluation for each model
for MODEL in "${MODELS[@]}"; do
    echo "========================================="
    echo "Evaluating model: $MODEL"
    echo "========================================="
    
    python Linear.py \
        --model $MODEL \
        --batch_size 128 \
        --num_workers 8 \
        --data_root ./data \
        --checkpoint ./resnet50_scalekd_e300.pth
    
    echo ""
done

echo "All evaluations completed!"