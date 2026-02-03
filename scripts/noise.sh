
cd ~/VLMtoresnet/ml-fastvit-main/

conda activate fastvit

python -m torch.distributed.launch --nproc_per_node=2 train_noise.py \
  /mnt/SSD2/ImageNet1k/ \
  --model fastvit_sa36 \
  --batch-size 32 \
  --noise-experiment \
  --initial-checkpoint Weights/fastvit_sa36.pth.tar \
  --resume Weights/fastvit_sa36.pth.tar \
  --noise-val-batches 200 \
  --noise-val-interval 2000 \
  --workers 10 \
  --log-wandb 