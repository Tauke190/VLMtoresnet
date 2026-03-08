
conda create --name fastvit python=3.9 -y
conda activate fastvit
python -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install einops shapely timm==1.0.15 yacs tensorboardX ftfy prettytable pymongo transformers diffdist seaborn vlkit inflect nltk umap-learn keyboard 
python -m pip install -U scikit-image keyboard ultralytics pycocotools scikit-learn packaging


# python -m pip install numpy==1.26.4
# python -m pip install albumentations==2.0.5
# python -c "import torch; print(torch.__version__);"
# python -c "import torch; print(torch.__version__);" #### 1.10
# python -c "import setuptools"
# python -m pip install setuptools==59.8.0




cd ~/VLMtoresnet/
# python Scripts/run.sh
conda activate fastvit


############## Training 
WT=logs/Weights/fastvit_sa36.pth.tar
IMAGENET_PATH=/mnt/SSD2/ImageNet1k/
VAL_SET="food101"
VAL_PATH=/mnt/SSD2/food-101
OUTPUT=./checkpoints
NUM_GPU=1
LOG=100
PORT=12351
# Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens, fastvit_sa36_lora_pp]
# Training methods = [default, baseline, distillation, attention_distillation]
# --method distillation



EXP=fastvit-adapter
MODEL=fastvit_sa36_adapter
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT train_baseline.py \
    $IMAGENET_PATH --model $MODEL --val-set $VAL_SET --validation-data-dir $VAL_PATH --log-interval $LOG \
    --validation-eval-interval 1 --initial-checkpoint $WT --output $OUTPUT --experiment $EXP \
    --freeze-backbone --native-amp --workers 12 \
    -b 32 --lr 1e-3 --drop-path 0.35 --mixup 0 --cutmix 0 --epochs 50 --input-size 3 224 224 
    


