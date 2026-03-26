conda activate fastvit

cd ~/VLMtoresnet/
# python Scripts/run.sh
conda activate fastvit

############## Training 
WT=logs/Weights/fastvit_sa36.pth.tar
IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
DIFFUSION_PATH=/mnt/SSD2/Diffision_images

DATASET_PATH=$IMAGENET_PATH

VAL_SET="food101"
VAL_PATH=/mnt/SSD2/food-101
NUM_GPU=2
LOG=100
PORT=12351

EXP=fastvit
MODEL=fastvit_sa36_lora_pp  # Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens, fastvit_sa36_lora_pp]
METHOD=distillation               # Training methods = [default, baseline, distillation, attention_distillation]
DATASET=imagenet              # [imagenet , diffusion]

OUTPUT=./checkpoints/${RUN_NAME}

RUN_NAME="${MODEL}_${METHOD}"

# Imagenet training
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT train_baseline.py \
    $DATASET_PATH --model $MODEL --dataset $DATASET --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP --method $METHOD \
    --freeze-backbone --native-amp --workers 12 --intra-epoch-val 20 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224

#  --log-wandb $RUN_NAME
#  --sanity-check

# #Diffusion training
# CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT train_baseline.py \
#     $DATASET_PATH --model $MODEL --dataset $DATASET --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
#     --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP --method $METHOD --num-classes 6000 \
#     --freeze-backbone --native-amp --workers 12 --log-wandb $RUN_NAME --sanity-check \
#     -b 16 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224