cd ~/VLMtoresnet/ml-fastvit-main/
# python Scripts/run.sh

conda activate fastvit

python validate.py /mnt/SSD2/ImageNet1k \
  --model fastvit_sa36_projector \
  --checkpoint Weights/fastvit_sa36.pth.tar \
  --eval-mode logits \
  --dataset fgvc_aircraft


# Eval mode [ logit , linearprobe , zeroshot]


# Imagenetpath : /mnt/SSD2/ImageNet1k
# fgvc-aircraft : /mnt/SSD2/fgvc-aircraft-2013b/data
# food101 : /mnt/SSD2/food-101
# cars : /mnt/SSD2/stanford_cars
# UCF : /mnt/SSD2/UCF101_midframes
# fer2013 : /mnt/SSD2/fer2013
# gtsrb : /mnt/SSD2/gtsrb
# sst2 : mnt/SSD2/rendered-sst2