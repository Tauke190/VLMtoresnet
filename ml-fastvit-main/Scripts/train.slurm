#!/bin/bash
#SBATCH --job-name=backbonefreeze
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
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
cd /home/av354855/VLMtoresnet/ml-fastvit-main

# Run unbuffered
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    /mnt/SSD2/ImageNet1k/ \
    --model fastvit_sa36_projector \
    --val-set "fgvc_aircraft" \
    --validation-data-dir /mnt/SSD2/fgvc-aircraft-2013b/data \
    --validation-eval-interval 1000 \
    --initial-checkpoint Weights/fastvit_sa36.pth.tar \
    --output ./checkpointsfreezebackbone \
    -b 64 --lr 1e-3 \
    --log-wandb --native-amp --input-size 3 256 256 \
    --drop-path 0.35 --mixup 0 --cutmix 0 \
    --experiment CLIPtoResNet \
    --workers 6 --epochs 50 \
    --freeze-backbone \
    --debug

# Optional cleanup
conda deactivate
