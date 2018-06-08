#!/bin/bash


ROOT="/home/popzq/workspace/models"
TFDET="${ROOT}/research/object_detection"
WORKSPACE="${ROOT}/workspace/det/ssd.head"
PIPELINE_CONFIG="${WORKSPACE}/ssd_mnetv2_coco.config"
TRAIN_DIR="${WORKSPACE}/train"

python ${TFDET}/eval.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --checkpoint_dir=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR}


