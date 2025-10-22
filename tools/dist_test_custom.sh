#!/usr/bin/env bash

CONFIG="configs/distillers/advanced_training_strategy/swin-l_distill_res50_img_s3_s4.py"
CHECKPOINT="resnet50_scalekd_e300.pth"
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher none ${@:1}