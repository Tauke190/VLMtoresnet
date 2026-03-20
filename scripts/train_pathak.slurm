#!/bin/bash

#SBATCH --job-name=E_3
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH --exclude=c4-4,c5-2
#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G 
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
NUM_GPU=2

PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"



if [[ "$SLURM_JOB_NODELIST" == "c4-4" ]]; then
        sbatch ucf_output/slurm-$SLURM_JOB_ID.sh
        exit 1
fi

FOOD_PATH=/home/c3-0/datasets/food-101
AIRCRAFT_PATH=/home/c3-0/datasets/data
IMAGENET_PATH=/home/c3-0/datasets/ImageNet
DIFF_PATH=/home/c3-0/datasets/Diffusion_gen_images/
DATASETS=(food101 aircraft imagenet diffusion)
DATASETS_PATHS=($FOOD_PATH $AIRCRAFT_PATH $IMAGENET_PATH $DIFF_PATH)
NUM_ELEMENTS=${#DATASETS[@]}


############## Training 
# VAL_PATH=/mnt/SSD2/food-101
# IMAGENET_PATH=/mnt/SSD2/ImageNet1k/

VAL_PATH=$FOOD_PATH
WT=logs/Weights/fastvit_sa36.pth.tar
VAL_SET="food101"
OUTPUT=./checkpoints
NUM_GPU=2
LOG=1000



####### Train ....  
# EXP=fastvit-adapter
# MODEL=fastvit_sa36_adapter
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT  train_baseline.py \
#     $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
#     --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
#     --freeze-backbone --native-amp --workers 8 \
#     -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 >> ucf_output/$EXP.txt


####### Eval .... 
EXP=fastvit-adapter
MODEL=fastvit_sa36_adapter

OUTPUT=ucf_output/$EXP-EVAL.txt 
FASTVIT_CKPT=checkpoints/$EXP/model_best_zeroshot.pth.tar
ls $FASTVIT_CKPT
# Loop through the indices of the arrays
for (( i=0; i<$NUM_ELEMENTS; i++ )); do
    Dataset="${DATASETS[$i]}"
    Path="${DATASETS_PATHS[$i]}"
    echo "Item: $Dataset, Paths: $Path"

    ########################
    # FastViT NECK PROBE
    ########################
    python eval_final.py --model $MODEL --dataset $Dataset --data-dir $Path \
      --model-checkpoint $FASTVIT_CKPT --feature-mode classification_neck >> $OUTPUT
done









  
    
# rsync -a ~/VLMtoresnet/scripts/train_pathak.slurm ucf0:~/VLMtoresnet/scripts/
# rsync -a ucf0:~/VLMtoresnet/ucf_output/ ~/VLMtoresnet/ucf_output/




# cd ~/VLMtoresnet/
# sbatch scripts/train_pathak.slurm
