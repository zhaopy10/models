#!/bin/bash


SLIM="/home/popzq/workspace/models/research/slim"
WORKSPACE="/home/popzq/workspace/models/workspace/cls"
DATASET_DIR="/home/popzq/disk/ILSVRC2012/validation"
VAL_DIR="${WORKSPACE}/train"

python ${SLIM}/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${VAL_DIR} \
    --eval_dir=${VAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=mobilenet_v2_020 \
    --eval_image_size=224 \
    --batch_size=1 \
    --use_gpu=False


