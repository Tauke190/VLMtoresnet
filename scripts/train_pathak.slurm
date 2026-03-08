#!/bin/bash

#SBATCH --job-name=E_1
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH --exclude=c4-4,c5-2
#SBATCH -C gmem48 --gres=gpu:2 --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=8

####SBATCH -c8
############## SBATCH -c8
################SBATCH -p gpu --qos=preempt
##################SBATCH -p gpu --qos=day
#################SBATCH -p gpu --qos=short 


# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash 


nvidia-smi
source ~/.bashrc
CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh ucf_output/
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh nm1:~/VLMtoresnet/ucf_output/


module load cuda/11.3
conda activate fastvit
cd ~/VLMtoresnet/


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

export DETECTRON2_DATASETS=/home/c3-0/datasets/flickr_dataset_30k/
export DATASET=/home/c3-0/datasets/flickr_dataset_30k/
export ADD_DATASET=/home/c3-0/datasets/
export TOKENIZERS_PARALLELISM=true 

NUM_GPU=2


if [[ "$SLURM_JOB_NODELIST" == "c4-4" ]]; then
        sbatch ucf_output/slurm-$SLURM_JOB_ID.sh
        exit 1
fi




############## Training 
# VAL_PATH=/mnt/SSD2/food-101
# IMAGENET_PATH=/mnt/SSD2/ImageNet1k/

VAL_PATH=/home/c3-0/datasets/food-101
IMAGENET_PATH=/home/c3-0/datasets/ImageNet

WT=logs/Weights/fastvit_sa36.pth.tar
VAL_SET="food101"
OUTPUT=./checkpoints
NUM_GPU=1
LOG=100




EXP=fastvit
MODEL=fastvit_sa36_adapter
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT  train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --method distillation \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 --sanity-check



# rsync -r nm1:~/VLMtoresnet/logs ~/VLMtoresnet/
# cd ~/VLMtoresnet/
# 
