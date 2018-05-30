#!/bin/bash

ROOT="/home/corp.owlii.com/yi.xu/workspace/models"
TFDET="${ROOT}/research/object_detection"
WORKSPACE="${ROOT}/workspace/det/pretrained"
PIPELINE_CONFIG="${WORKSPACE}/pipeline.config"
TRAIN_DIR="${WORKSPACE}/train"
HOME="/home/corp.owlii.com/yi.xu"
TFBAZEL="${HOME}/tensorflow/tensorflow/bazel-bin"

CKPT="${WORKSPACE}/coco_ckpt/model.ckpt-34205"
DEPLOY="${TRAIN_DIR}/deploy"
mkdir -p ${DEPLOY}

#pb_path="${DEPLOY}/saved_model/saved_model.pb"
#log_dir="${DEPLOY}/saved_model/"
#pbtxt_name="saved_model.pbtxt"
pb_path="${DEPLOY}/frozen_inference_graph.pb"
pbtxt_name="frozen_inference_graph.pbtxt"

input_node="image_tensor:0"
output_node="detection_boxes:0,detection_scores:0,detection_classes:0,num_detections:0"

######################################
python ${TFDET}/export_inference_graph.py \
    --input_type="image_tensor" \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix=${CKPT} \
    --output_directory=${DEPLOY}

python ${WORKSPACE}/../my_graph_utils/pb2pbtxt.py \
    --pb_path=${pb_path} \
    --log_dir=${DEPLOY} \
    --pbtxt_name=${pbtxt_name}


#output_node="Squeeze:0,concat_1:0"
#ml_output_node=${output_node}
output_node="detection_boxes:0,detection_scores:0,detection_classes:0"
ml_output_node='all'

${TFBAZEL}/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${DEPLOY}/frozen_inference_graph.pb \
    --out_graph=${DEPLOY}/deploy_graph.pb \
    --inputs=${input_node} \
    --outputs=${output_node} \
    --transforms='strip_unused_nodes(type=float, shape="1,300,300,3")
                  remove_nodes(op=Identity, op=CheckNumerics)
                  fold_constants(ignore_errors=true)
                  fold_batch_norms
                  fold_old_batch_norms'

python ${WORKSPACE}/../my_graph_utils/pb2pbtxt.py \
    --pb_path=${DEPLOY}/deploy_graph.pb \
    --log_dir=${DEPLOY} \
    --pbtxt_name=${DEPLOY}/deploy_graph.pbtxt

python ${WORKSPACE}/../tf_coreml_utils/tf2coreml.py \
    --input_pb_file=${DEPLOY}/deploy_graph.pb \
    --output_mlmodel=${DEPLOY}/deploy_graph.mlmodel \
    --input_node_name=${input_node} \
    --output_node_name=${ml_output_node}




