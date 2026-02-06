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


python -m torch.distributed.launch --nproc_per_node=2 train_noise.py \
  /home/c3-0/datasets/ImageNet \
  --model fastvit_sa36 \
  --batch-size 128 \
  --native-amp \
  --noise-experiment


python train_noise.py \
  /home/c3-0/datasets/ImageNet \
  --model fastvit_sa36 \
  --batch-size 128 \
  --native-amp \
  --noise-experiment \
  --resume fastvit_sa36.pth.tar


python validate.py /home/c3-0/datasets/ImageNet --model fastvit_sa36 \
--checkpoint /home/av354855/VLMtoresnet/ml-fastvit-main/checkpoints/CLIPtoResNet/model_best.pth.tar  \

python zeroshot_eval.py \
  --model fastvit_sa36 \
  --zeroshot-dataset aircraft \
  --zeroshot-data-dir /home/c3-0/datasets/fgvc-aircraft-2013b/data \
  --model-checkpoint checkpoints/CLIPtoResNet/model_best_aircraft.pth.tar \
  --projector-checkpoint checkpoints/CLIPtoResNet/projector_best_aircraft.pth.tar \
  --batch-size 128


python zeroshot_eval.py \
  --model fastvit_sa36 \
  --zeroshot-dataset imagenet \
  --zeroshot-data-dir /home/c3-0/datasets/ImageNet \
  --model-checkpoint checkpoints/CLIPtoResNet/model_best_aircraft.pth.tar \
  --projector-checkpoint checkpoints/CLIPtoResNet/projector_best_aircraft.pth.tar \
  --batch-size 128

  python zeroshot_eval.py \
  --model fastvit_sa36 \
  --zeroshot-dataset oxfordpet \
  --zeroshot-data-dir /home/c3-0/datasets/oxford_pets \
  --model-checkpoint checkpoints/CLIPtoResNet/model_best_aircraft.pth.tar \
  --projector-checkpoint checkpoints/CLIPtoResNet/projector_best_aircraft.pth.tar \
  --batch-size 128

   python zeroshot_eval.py \
  --model fastvit_sa36 \
  --zeroshot-dataset food101 \
  --zeroshot-data-dir /home/c3-0/datasets/food-101 \
  --model-checkpoint checkpoints/CLIPtoResNet/model_best_aircraft.pth.tar \
  --projector-checkpoint checkpoints/CLIPtoResNet/projector_best_aircraft.pth.tar \
  --batch-size 128



