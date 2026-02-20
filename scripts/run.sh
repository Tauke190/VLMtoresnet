

cd /home/av354855/projects/VLMtoresnet/
# python Scripts/run.sh
conda activate fastvit


############## Training 
WT=logs/Weights/fastvit_sa36.pth.tar
# WT=/home/av354855/projects/VLMtoresnet/checkpoints/fastvit-adapter/model_best_zeroshot.pth.tar
IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
VAL_SET="food101"
VAL_PATH=/mnt/SSD2/food-101
OUTPUT=./checkpoints
NUM_GPU=2
LOG=100

# Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens, fastvit_sa36_lora_pp]
# Training methods = [default, baseline, distillation, attention_distillation]

EXP=fastvit
MODEL=fastvit_sa36_lora_pp
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --method attention_distillation \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 

conda deactivate