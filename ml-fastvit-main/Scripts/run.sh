
cd ~/VLMtoresnet/ml-fastvit-main/
# python Scripts/run.sh
conda activate fastvit

NUM_GPU=1
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    /mnt/SSD2/ImageNet1k/ \
    --model fastvit_sa36_projector \
    --val-set "food101" \
    --validation-data-dir /mnt/SSD2/food-101 \
    --validation-eval-interval 2000 \
    --initial-checkpoint Weights/fastvit_sa36.pth.tar \
    --output ./checkpointsfreezebackbone \
    -b 64 --lr 1e-3 \
    --log-wandb --native-amp --input-size 3 256 256 \
    --drop-path 0.35 --mixup 0 --cutmix 0 \
    --workers 10 --epochs 50 \
    --freeze-backbone 
    # --log-wandb --experiment CLIPtoResNet \
    

# Initialized Aircraft zero-shot evaluation with 100 classes.

# Evaluating.... 14 Iteratiopns on a B=256, with 3584 datapoints

# Aircraft zero-shot before training: Acc@1 = 0.78%
# Aircraft zero-shot before training: Acc@5 = 4.44%


# Training.... 5004 Iteratiopns on a B=256, with 1 281 024 datapoints

# Training.... 2502 Iteratiopns on a B=256, with 640 512 datapoints                                                                                                                     
# Training.... 2502 Iteratiopns on a B=256, with 640 512 datapoints 