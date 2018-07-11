#!/bin/bash

ROOT="/home/corp.owlii.com/yi.xu/workspace/models"
TFDET="${ROOT}/research/object_detection"
WORKSPACE="${ROOT}/workspace/det/frm1.100"
PIPELINE_CONFIG="${WORKSPACE}/faster_rcnn_mobilenet_v1_100_coco.config"
TRAIN_DIR="${WORKSPACE}/train"

python ${TFDET}/train.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --train_dir=${TRAIN_DIR}

