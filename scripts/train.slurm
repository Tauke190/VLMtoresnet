#!/bin/bash
#SBATCH --job-name=atn_distill_lora
#SBATCH --output=ucf_output/atn_distill_lora-%j.out
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

# Set UTF-8 encoding to handle Unicode characters
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8

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
IMAGENET_PATH=/home/c3-0/datasets/ImageNet
VAL_SET="food101"
VAL_PATH=/home/c3-0/datasets/food-101
OUTPUT=./checkpoints
NUM_GPU=2
LOG=500


EXP=fastvit
MODEL=fastvit_sa36_lora_pp # Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens, fastvit_sa36_lora_pp]
METHOD=distillation # Training methods = [default, baseline, distillation, attention_distillation]
RUN_NAME="${MODEL}_${METHOD}"
OUTPUT=./checkpoints/${RUN_NAME}

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --method $METHOD --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --log-wandb $RUN_NAME \
    -b 64 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224

conda deactivate