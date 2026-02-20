#!/bin/bash

TARGET_DIR="/mnt/SSD2/posetrack/posetrack_2018"
mkdir -p "$TARGET_DIR"

base_url="https://s3.app.hyper.ai/Dataset-Upload/datasets/3AG3O6qSLEN/1/data/PoseTrack2018"
params="?response-content-disposition=attachment%3B%20filename%3D%22posetrack18_images.tar.%s%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260215T152352Z&X-Amz-SignedHeaders=host&X-Amz-Expires=14399&X-Amz-Credential=minioadmin%2F20260215%2F%2Fs3%2Faws4_request&X-Amz-Signature=e5ecd39c7f232bb3bb6fb1c81cb3e9ff1f1c61bc09f5935f2dfc8f4ea3dc8090"

for suffix in ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar; do
    url="${base_url}/posetrack18_images.tar.${suffix}$(printf "$params" "$suffix")"
    wget "$url" -O "$TARGET_DIR/posetrack18_images.tar.${suffix}"
done