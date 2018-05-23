#!/bin/bash

ROOT="/home/corp.owlii.com/yi.xu/workspace/models"
TFDET="${ROOT}/research/object_detection"
PIPELINE_CONFIG="${TFDET}/samples/configs/ssdlite_mobilenet_v2_coco.config"
WORKSPACE="${ROOT}/workspace/det"
TRAIN_DIR="${WORKSPACE}/train"

CKPT="${TRAIN_DIR}/model.ckpt-123164"
DEPLOY="${TRAIN_DIR}/deploy"
mkdir -p ${DEPLOY}

#pb_path="${DEPLOY}/saved_model/saved_model.pb"
#log_dir="${DEPLOY}/saved_model/"
#pbtxt_name="saved_model.pbtxt"
pb_path="${DEPLOY}/frozen_inference_graph.pb"
log_dir="${DEPLOY}"
pbtxt_name="frozen_inference_graph.pbtxt"


######################################

python ${TFDET}/export_inference_graph.py \
    --input_type="image_tensor" \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix=${CKPT} \
    --output_directory=${DEPLOY}


python ${WORKSPACE}/my_graph_utils/pb2pbtxt.py \
    --pb_path=${pb_path} \
    --log_dir=${log_dir} \
    --pbtxt_name=${pbtxt_name}



