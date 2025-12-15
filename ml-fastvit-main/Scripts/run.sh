



cd ~/VLMtoresnet/ml-fastvit-main/
# python Scripts/run.sh
conda activate fastvit


NUM_GPU=1
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train_baseline.py \
    /mnt/SSD2/imagenet/ \
    --model fastvit_sa36_projector \
    --aircraft-data-dir /mnt/SSD2/fgvc-aircraft-2013b/data \
    --aircraft-eval-interval 1000 \
    --initial-checkpoint Weights/fastvit_sa36.pth.tar \
    --output ./checkpoints \
    -b 256 --lr 1e-3 \
    --log-wandb --native-amp --input-size 3 256 256 \
    --drop-path 0.35 --mixup 0 --cutmix 0 \
    --experiment CLIPtoResNet \
    --workers 4 --epochs 50 \
    --freeze-backbone






