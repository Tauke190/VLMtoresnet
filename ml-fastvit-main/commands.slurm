#!/bin/bash

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