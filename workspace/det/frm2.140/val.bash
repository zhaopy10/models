#!/bin/bash


ROOT="/home/corp.owlii.com/yi.xu/workspace/models"
TFDET="${ROOT}/research/object_detection"
WORKSPACE="${ROOT}/workspace/det/frm2.140"
PIPELINE_CONFIG="${WORKSPACE}/faster_rcnn_mobilenet_v2_140_coco.config"
TRAIN_DIR="${WORKSPACE}/train"

python ${TFDET}/eval.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --checkpoint_dir=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR}


