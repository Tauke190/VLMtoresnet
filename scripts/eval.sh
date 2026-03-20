# # python Scripts/run.sh
# conda activate fastvit

#  #[ Models - [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lora_pp , fastvit_sa36_lora_pp]

# # --- Configuration ---
# MODEL=fastvit_sa36_lora_pp
# CHECKPOINT=checkpoints/fastvit-lora-pp/lora_contrastive_distillation.pth.tar
# EVAL_MODE=zeroshot   # [ logit , linearprobe , zeroshot ]

# # --- Dataset name -> path mapping ---
# declare -A DATASET_PATHS=(
#     [imagenet1k]=/mnt/SSD2/ImageNet1k
#     [fgvc_aircraft]=/mnt/SSD2/fgvc-aircraft-2013b/data
#     [food101]=/mnt/SSD2/food-101
#     [cars]=/mnt/SSD2/stanford_cars
#     [ucf101]=/mnt/SSD2/UCF101_midframes
#     [fer2013]=/mnt/SSD2/fer2013
#     [gtsrb]=/mnt/SSD2/gtsrb
#     [sst2]=/mnt/SSD2/rendered-sst2
# )

# DATASETS=(food101 ucf101 imagenet1k)

# # --- Loop over datasets ---
# for DS in "${DATASETS[@]}"; do
#     DATA_PATH="${DATASET_PATHS[$DS]}"
#     if [ -z "$DATA_PATH" ]; then
#         echo "ERROR: No path defined for dataset '$DS'. Skipping."
#         continue
#     fi
#     echo "============================================"
#     echo "Evaluating: $DS  ($DATA_PATH)"
#     echo "============================================"
#     python validate.py "$DATA_PATH" \
#         --model "$MODEL" \
#         --checkpoint "$CHECKPOINT" \
#         --eval-mode "$EVAL_MODE" \
#         --dataset "$DS"
# done


# Imagenetpath : /mnt/SSD2/ImageNet1k
# fgvc-aircraft : /mnt/SSD2/fgvc-aircraft-2013b/data
# food101 : /mnt/SSD2/food-101
# cars : /mnt/SSD2/stanford_cars
# UCF : /mnt/SSD2/UCF101_midframes
# fer2013 : /mnt/SSD2/fer2013
# gtsrb : /mnt/SSD2/gtsrb
# sst2 : mnt/SSD2/rendered-sst2

FOOD_PATH=/mnt/SSD2/food-101
IMAGENET_PATH=/mnt/SSD2/ImageNet1k
AIRCRAFT_PATH=/mnt/SSD2/fgvc-aircraft-2013b/data

DATASET=food101 # imagenet , food101 , aircraft

DATASET_PATH=$FOOD_PATH
FASTVIT_MODEL=fastvit_sa36_lrtokens # Model = [fastvit_sa36, fastvit_sa36_adapter, fastvit_sa36_lrtokens, fastvit_sa36_lora_pp]
FASTVIT_CKPT=checkpoints/fastvit_lrtokens/fastvit_sa36_lrtokens_distillation.pth.tar

# python eval_final.py --feature-mode zeroshot --model $FASTVIT_MODEL --dataset $DATASET --data-dir $DATASET_PATH --model-checkpoint $FASTVIT_CKPT


python eval_final.py --model $FASTVIT_MODEL --dataset $DATASET --data-dir $DATASET_PATH --model-checkpoint $FASTVIT_CKPT --feature-mode backbone1 
