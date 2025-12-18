



cd ~/VLMtoresnet/ml-fastvit-main/
# python Scripts/run.sh
conda activate fastvit


NUM_GPU=2
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    /mnt/SSD2/ImageNet1k/ \
    --model fastvit_sa36_projector \
    --val-set "fgvc_aircraft" \
    --validation-data-dir /mnt/SSD2/fgvc-aircraft-2013b/data \
    --validation-eval-interval 1000 \
    --initial-checkpoint Weights/fastvit_sa36.pth.tar \
    --output ./checkpoints \
    -b 64 --lr 1e-3 \
    --log-wandb --native-amp --input-size 3 224 224 \
    --drop-path 0.35 --mixup 0 --cutmix 0 \
    --experiment CLIPtoResNet \
    --workers 6 --epochs 50 \
    --freeze-backbone \
    --debug

# Initialized Aircraft zero-shot evaluation with 100 classes.

# Evaluating.... 14 Iteratiopns on a B=256, with 3584 datapoints
# Aircraft zero-shot before training: Acc@1 = 1.14%
# Aircraft zero-shot before training: Acc@5 = 5.50%

# Evaluating.... 7 Iteratiopns on a B=256, with 1792 datapoints
# <class 'models.proposed_model.FastViT_projector'>
# Aircraft zero-shot before training: Acc@1 = 0.95%
# Aircraft zero-shot before training: Acc@5 = 5.08%

# Training.... 5004 Iteratiopns on a B=256, with 1 281 024 datapoints

# Training.... 2502 Iteratiopns on a B=256, with 640 512 datapoints                                                                                                                     
# Training.... 2502 Iteratiopns on a B=256, with 640 512 datapoints 
