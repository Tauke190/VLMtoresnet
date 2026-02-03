#!/bin/bash

cd ~/projects/VLMtoresnet/
conda activate fastvit

############## CCI Visualization Settings
MODEL="ViT-B/32"          # CLIP model: ViT-B/32, ViT-B/16, ViT-L/14
N_CLUSTERS=7              # Number of clusters for K-means
OUTPUT_DIR="./cci_outputs"

# Heatmap settings
INTERPOLATION="bicubic"   # Options: bicubic, spline, gaussian, lanczos
COLORMAP="jet"            # Options: jet, hot, viridis, inferno, plasma, etc.
ALPHA=0.5                 # Overlay transparency (0-1)

# Example 1: Single image with text prompt
IMAGE_PATH="/home/av354855/projects/VLMtoresnet/dog.jpeg"
TEXT_PROMPT="a photo of a dog"

python run_cci.py \
    --image "$IMAGE_PATH" \
    --text "$TEXT_PROMPT" \
    --model "$MODEL" \
    --n-clusters $N_CLUSTERS \
    --output-dir "$OUTPUT_DIR" \
    --interpolation "$INTERPOLATION" \
    --colormap "$COLORMAP" \
    --alpha $ALPHA \
    --save-detailed

# Example 2: Batch processing a directory
# python run_cci.py \
#     --image-dir "/path/to/images/" \
#     --text "a photo of an animal" \
#     --model "$MODEL" \
#     --n-clusters $N_CLUSTERS \
#     --output-dir "$OUTPUT_DIR" \
#     --interpolation spline \
#     --colormap viridis

# Example 3: Multiple text prompts for comparison
# python run_cci.py \
#     --image "$IMAGE_PATH" \
#     --text "a dog" "an animal" "a pet" \
#     --model "$MODEL" \
#     --n-clusters $N_CLUSTERS \
#     --output-dir "$OUTPUT_DIR" \
#     --interpolation gaussian \
#     --colormap hot \
#     --compare-prompts
