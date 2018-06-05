#!/bin/bash

ROOT="/home/xuyithu/workspace/models"
TFDET="${ROOT}/research/object_detection"
WORKSPACE="${ROOT}/workspace/det/coco.person"
PIPELINE_CONFIG="${WORKSPACE}/ssd_mnetv2_coco.config"
TRAIN_DIR="${WORKSPACE}/train"

num_clones=4

python ${TFDET}/train.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --train_dir=${TRAIN_DIR} \
    --num_clones=${num_clones}


