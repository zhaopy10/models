#!/bin/bash


ROOT="/home/corp.owlii.com/yi.xu/workspace/models"
TFDET="${ROOT}/research/object_detection"
#PIPELINE_CONFIG="${TFDET}/samples/configs/ssdlite_mobilenet_v2_coco.config"
WORKSPACE="${ROOT}/workspace/det/oid"
PIPELINE_CONFIG="${WORKSPACE}/pipeline.config"
TRAIN_DIR="${WORKSPACE}/train"

python ${TFDET}/eval.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --checkpoint_dir=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR}


