#!/bin/bash

################################################################################################################
############## Training 
WT=logs/fastvit_sa36.pth.tar
IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
EXP=CLIPtoResNet
VAL_SET="food101"
VAL_PATH=/mnt/SSD2/food-101
OUTPUT=./checkpoints
LOG=500

############## Baseline Training (FastViT) E2E
EXP=fastvit-vanilla
MODEL=fastvit_sa36
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP --method 'default' \
    --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 --vanilla-eval
# --freeze-backbone ## wont do anything 
# --val-set $VAL_SET ## cant do cross validation on vanilla model so it wont do anything    


##### Projector (frozen backbone)
EXP=fastvit-proj
MODEL=fastvit_sa36_projector
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
  # --freeze-backbone # remove to train entire model 


##### Lr-TK0 (freeze the backbone )
EXP=fastvit-lrtk0
MODEL=fastvit_sa36_lrtokens
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    

##### Adapter
EXP=fastvit-adapter
MODEL=fastvit_sa36_adapter
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    


################################################################################################################
############## Evalaution
################################################################################################################

FASTVIT_MODEL=fastvit_sa36
FOOD_PATH=/mnt/SSD2/food-101
AIRCRAFT_PATH=/mnt/SSD2/fgvc-aircraft-2013b/data
IMAGENET_PATH=/mnt/SSD2/ImageNet1k
DIFF_PATH=/mnt/SSD2/Diffision_images

FASTVIT_CKPT=logs/fastvit_sa36.pth.tar

FASTVIT_CKPT=checkpoints/CLIPtoResNet/model_best.pth.tar
PROJECTOR_CKPT=checkpoints/CLIPtoResNet/projector_best.pth.tar



### Food 101
DATASET=food101 
DATASET_PATH=$FOOD_PATH

### aircraft
DATASET=aircraft 
DATASET_PATH=$AIRCRAFT_PATH

### imagenet
DATASET=imagenet 
DATASET_PATH=$IMAGENET_PATH

### diffusion
DATASET=diffusion 
DATASET_PATH=$DIFF_PATH

########################
# CLIP ZERO SHOT (Text Based)
########################
python zeroshot_eval.py --model clip --zeroshot-dataset $DATASET --zeroshot-data-dir $DATASET_PATH

########################
# FastViT BACKBONE PROBE
########################
python eval_final.py --model $FASTVIT_MODEL --dataset $DATASET --data-dir $DATASET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode backbone1

########################
# FastViT NECK PROBE
########################
python eval_final.py --model $FASTVIT_MODEL --dataset $DATASET --data-dir $DATASET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classification_neck

########################
# FastViT CLASSIFIER LOGITS
########################
python eval_final.py --model $FASTVIT_MODEL --dataset $DATASET --data-dir $DATASET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classifier --eval-mode logits























##### This needs to be fixed, will come to this later..... (model name is wrong )
########################
# FastViT ZERO SHOT (PROJECTOR) 
########################

python zeroshot_eval.py \
--model fastvit_sa36 \
--zeroshot-dataset food101 \
--zeroshot-data-dir $FOOD_PATH \
--model-checkpoint $FASTVIT_CKPT \
--projector-checkpoint $PROJECTOR_CKPT

python zeroshot_eval.py \
--model fastvit_sa36 \
--zeroshot-dataset aircraft \
--zeroshot-data-dir $AIRCRAFT_PATH \
--model-checkpoint $FASTVIT_CKPT \
--projector-checkpoint $PROJECTOR_CKPT

python zeroshot_eval.py \
--model fastvit_sa36 \
--zeroshot-dataset imagenet \
--zeroshot-data-dir $IMAGENET_PATH \
--model-checkpoint $FASTVIT_CKPT \
--projector-checkpoint $PROJECTOR_CKPT

python zeroshot_eval.py \
--model fastvit_sa36 \
--zeroshot-dataset diffusion \
--zeroshot-data-dir $DIFF_PATH \
--model-checkpoint $FASTVIT_CKPT \
--projector-checkpoint $PROJECTOR_CKPT




