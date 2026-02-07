# python Scripts/run.sh
conda activate fastvit

 #[ Models - [fastvit_sa36 fastvit_sa36_adapter , fastvit_sa36_lrtokens]

python validate.py /mnt/SSD2/ImageNet1k \
  --model fastvit_sa36_lrtokens\
  --checkpoint checkpoints/fastvit_lrtokens/model_best_zeroshot.pth.tar \
  --eval-mode linearprobe \
  --dataset imagenet1k


# Eval mode [ logit , linearprobe , zeroshot]


# Imagenetpath : /mnt/SSD2/ImageNet1k
# fgvc-aircraft : /mnt/SSD2/fgvc-aircraft-2013b/data
# food101 : /mnt/SSD2/food-101
# cars : /mnt/SSD2/stanford_cars
# UCF : /mnt/SSD2/UCF101_midframes
# fer2013 : /mnt/SSD2/fer2013
# gtsrb : /mnt/SSD2/gtsrb
# sst2 : mnt/SSD2/rendered-sst2