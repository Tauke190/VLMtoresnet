# python Scripts/run.sh
conda activate fastvit

 #[ Models - [fastvit_sa36 fastvit_sa36_adapter , fastvit_sa36_lrtokens]

# --- Configuration ---
MODEL=fastvit_sa36_adapter
CHECKPOINT=checkpoints/fastvit-adapter/model_best_zeroshot.pth.tar
EVAL_MODE=linearprobe   # [ logit , linearprobe , zeroshot ]

# --- Dataset name -> path mapping ---
declare -A DATASET_PATHS=(
    [imagenet1k]=/mnt/SSD2/ImageNet1k
    [fgvc_aircraft]=/mnt/SSD2/fgvc-aircraft-2013b/data
    [food101]=/mnt/SSD2/food-101
    [cars]=/mnt/SSD2/stanford_cars
    [ucf101]=/mnt/SSD2/UCF101_midframes
    [fer2013]=/mnt/SSD2/fer2013
    [gtsrb]=/mnt/SSD2/gtsrb
    [sst2]=/mnt/SSD2/rendered-sst2
)

# --- Datasets to evaluate (subset of the keys above) ---
DATASETS=(food101 ucf101 imagenet1k)

# --- Loop over datasets ---
for DS in "${DATASETS[@]}"; do
    DATA_PATH="${DATASET_PATHS[$DS]}"
    if [ -z "$DATA_PATH" ]; then
        echo "ERROR: No path defined for dataset '$DS'. Skipping."
        continue
    fi
    echo "============================================"
    echo "Evaluating: $DS  ($DATA_PATH)"
    echo "============================================"
    python validate.py "$DATA_PATH" \
        --model "$MODEL" \
        --checkpoint "$CHECKPOINT" \
        --eval-mode "$EVAL_MODE" \
        --dataset "$DS"
done


# Imagenetpath : /mnt/SSD2/ImageNet1k
# fgvc-aircraft : /mnt/SSD2/fgvc-aircraft-2013b/data
# food101 : /mnt/SSD2/food-101
# cars : /mnt/SSD2/stanford_cars
# UCF : /mnt/SSD2/UCF101_midframes
# fer2013 : /mnt/SSD2/fer2013
# gtsrb : /mnt/SSD2/gtsrb
# sst2 : mnt/SSD2/rendered-sst2