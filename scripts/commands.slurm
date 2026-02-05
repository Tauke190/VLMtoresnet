#!/bin/bash

################################################################################################################
############## Training 
WT=logs/fastvit_sa36.pth.tar
IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
EXP=CLIPtoResNet
VAL_SET="food101"
VAL_PATH=/mnt/SSD2/food-101
OUTPUT=./checkpoints

############## Baseline Training (FastViT) (frozen backbone)
MODEL=fastvit_sa36
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --validation-eval-interval 2000 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    
##### Projector
MODEL=fastvit_sa36_lrtokens
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --validation-eval-interval 2000 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --clip-loss-weight 1 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    

##### Lr-TK0
MODEL=fastvit_sa36_lrtokens
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --validation-eval-interval 2000 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --clip-loss-weight 1 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    
##### Adapter
MODEL=fastvit_sa36_adapter
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH \
    --validation-eval-interval 2000 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 --clip-loss-weight 1 \
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


################################################################################################################
############## ADDED RUNS (CLIP + FastViT BENCHMARK GRID)
################################################################################################################

FOOD_PATH=/mnt/SSD2/food-101
AIRCRAFT_PATH=/mnt/SSD2/fgvc-aircraft-2013b/data
IMAGENET_PATH=/mnt/SSD2/ImageNet1k
DIFF_PATH=/mnt/SSD2/Diffision_images

FASTVIT_CKPT=checkpoints/CLIPtoResNet/model_best.pth.tar
PROJECTOR_CKPT=checkpoints/CLIPtoResNet/projector_best.pth.tar


############## CLIP ZERO SHOT
python clip_zeroshot_eval.py --dataset food101 --data-dir $FOOD_PATH
python clip_zeroshot_eval.py --dataset aircraft --data-dir $AIRCRAFT_PATH
python clip_zeroshot_eval.py --dataset imagenet --data-dir $IMAGENET_PATH
python clip_zeroshot_eval.py --dataset diffusion --data-dir $DIFF_PATH


############## CLIP LINEAR PROBE
python clip_linear_probe.py --dataset food101 --data-dir $FOOD_PATH
python clip_linear_probe.py --dataset aircraft --data-dir $AIRCRAFT_PATH
python clip_linear_probe.py --dataset imagenet --data-dir $IMAGENET_PATH
python clip_linear_probe.py --dataset diffusion --data-dir $DIFF_PATH


############## FastViT LINEAR PROBE — BACKBONE
python linear_probe.py --model fastvit_sa36 --dataset food101 --data-dir $FOOD_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode backbone1
python linear_probe.py --model fastvit_sa36 --dataset aircraft --data-dir $AIRCRAFT_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode backbone1
python linear_probe.py --model fastvit_sa36 --dataset imagenet --data-dir $IMAGENET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode backbone1
python linear_probe.py --model fastvit_sa36 --dataset diffusion --data-dir $DIFF_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode backbone1


############## FastViT LINEAR PROBE — NECK
python linear_probe.py --model fastvit_sa36 --dataset food101 --data-dir $FOOD_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classification_neck
python linear_probe.py --model fastvit_sa36 --dataset aircraft --data-dir $AIRCRAFT_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classification_neck
python linear_probe.py --model fastvit_sa36 --dataset imagenet --data-dir $IMAGENET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classification_neck
python linear_probe.py --model fastvit_sa36 --dataset diffusion --data-dir $DIFF_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classification_neck


############## FastViT CLASSIFIER LOGITS (ImageNet)
python linear_probe.py --model fastvit_sa36 --dataset imagenet --data-dir $IMAGENET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode classifier --eval-mode logits


############## FastViT ZERO-SHOT (PROJECTOR REQUIRED)
python zeroshot_eval.py --model fastvit_sa36 --zeroshot-dataset food101 --zeroshot-data-dir $FOOD_PATH --model-checkpoint $FASTVIT_CKPT --projector-checkpoint $PROJECTOR_CKPT
python zeroshot_eval.py --model fastvit_sa36 --zeroshot-dataset aircraft --zeroshot-data-dir $AIRCRAFT_PATH --model-checkpoint $FASTVIT_CKPT --projector-checkpoint $PROJECTOR_CKPT
python zeroshot_eval.py --model fastvit_sa36 --zeroshot-dataset imagenet --zeroshot-data-dir $IMAGENET_PATH --model-checkpoint $FASTVIT_CKPT --projector-checkpoint $PROJECTOR_CKPT
python zeroshot_eval.py --model fastvit_sa36 --zeroshot-dataset diffusion --zeroshot-data-dir $DIFF_PATH --model-checkpoint $FASTVIT_CKPT --projector-checkpoint $PROJECTOR_CKPT
