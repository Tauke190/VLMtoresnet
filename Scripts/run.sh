

cd ~/VLMtoresnet/
# python Scripts/run.sh
conda activate fastvit


############## Training 
WT=logs/fastvit_sa36.pth.tar
IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
EXP=CLIPtoResNet
VAL_SET="food101"
VAL_PATH=/mnt/SSD2/food-101
OUTPUT=./checkpoints




MODEL=fastvit_sa36_adapter
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --validation-eval-interval 2000 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --clip-loss-weight 1 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    


# --checkpoint-name fastvitsa36_projector_lrtokens.pth.tar \       # 
# --log-wandb --experiment CLIPtoResNet \
# --debug



--