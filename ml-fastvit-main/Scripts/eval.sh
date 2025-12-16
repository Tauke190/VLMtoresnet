cd ~/VLMtoresnet/ml-fastvit-main/
# python Scripts/run.sh


conda activate fastvit
python validate.py /mnt/SSD2/food-101\
  --model fastvit_sa36_projector \
  --checkpoint Weights/fastvit_sa36.pth.tar \
  --eval-mode linearprobe \
  --linearprobe-dataset food101

# Imagenetpath : /mnt/SSD2/ImageNet1k