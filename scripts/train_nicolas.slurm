#!/bin/bash
#SBATCH --job-name=sa36_lrtokens
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=12


# #SBATCH -C gmem48 gmem32 gmem24 gmem16

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

IMAGENET_PATH=/home/c3-0/datasets/ImageNet
DIFF_PATH=/home/c3-0/datasets/Diffusion_gen_images/
VAL_PATH=/home/c3-0/datasets/food-101
VAL_SET="food101"
NUM_GPU=2
WT=Weights/fastvit_sa36.pth.tar

# Change directory
cd ~/ 

######## Dont change anything above (except GPUs)

#### Run Projector model on Difussion Dataset (Frozen backbone )
#### Run Projector model on Difussion Dataset (E2E )

EXP=fastvit
OUTPUT=./checkpoints
MODEL=fastvit_sa36_adapter

OUTPUT_TXT=ucf_output/$EXP.txt

# Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens]
# Training methods = [default, baseline, distillation]


# ##### Projector (frozen backbone)
# EXP=fastvit-proj
# MODEL=fastvit_sa36_projector
# CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
#     $DIFF_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
#     --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
#     --freeze-backbone --native-amp --workers 12 \
#     -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 >> ucf_output/name_of_exp.txt
#   # --freeze-backbone # remove to train entire model 

## E2E projector model 
EXP=fastvit-proj
MODEL=fastvit_sa36_projector
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $DIFF_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 >> ucf_output/name_of_exp.txt
  # --freeze-backbone # remove to train entire model 



conda deactivate

rsync -a ucf_output/* nm1:~/robustness_object_detection/GLIP/ucf_output/

# cd ~/VLMtoresnet/
# sbatch scripts/train_nicolas.slurm
# scancel <job id>