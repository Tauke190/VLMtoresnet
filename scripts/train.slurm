#!/bin/bash
#SBATCH --job-name=lrtoken_fastvit_distill
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=12

PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

# Load necessary modules
module load anaconda3
module load cuda

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
cd /home/av354855/VLMtoresnet

# Run unbuffered

WT=Weights/fastvit_sa36.pth.tar
# WT=/home/av354855/projects/VLMtoresnet/checkpoints/fastvit-adapter/model_best_zeroshot.pth.tar
IMAGENET_PATH=/home/c3-0/datasets/ImageNet
EXP=fastvit
VAL_SET="food101"
VAL_PATH=/home/c3-0/datasets/food-101
OUTPUT=./fastvit_adapter_checkpoints
NUM_GPU=2
MODEL=fastvit_sa36_lrtokens

# Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens]
# Training methods = [default, baseline, distillation]

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --method distillation --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --checkpoint-name $MODEL \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 --log-wandb
conda deactivate