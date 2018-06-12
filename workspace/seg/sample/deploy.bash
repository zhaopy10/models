#!/bin/bash


project="sample"
log_dir="train"

#HOME="/home/corp.owlii.com/yi.xu"
TFBAZEL="${HOME}/tensorflow/tensorflow/bazel-bin"
ROOT="${HOME}/workspace/models/workspace/seg"
WORKSPACE="${ROOT}/${project}"
TRAIN_UTILS="${ROOT}/my_training_utils"
PYTHONPATH="${PYTHONPATH}:${WORKSPACE}:${WORKSPACE}/.."
DEPLOY_DIR="${WORKSPACE}/${log_dir}/deploy"
mkdir -p ${DEPLOY_DIR}

ckpt="${WORKSPACE}/${log_dir}/model.ckpt-12"
model_name="mobilenet_v2_035_sgmt"
frozen_name="frozen_graph"
deploy_name="deploy_graph"
input_node="image:0"
output_node="heatmap:0"
input_shape="1,512,512,3"
output_size="512,512"
num_classes=2


python ${TRAIN_UTILS}/deploy.py \
    --trained_checkpoint_prefix=${ckpt} \
    --output_directory=${DEPLOY_DIR} \
    --output_graph_name="${frozen_name}.pb" \
    --input_shape=${input_shape} \
    --output_size=${output_size} \
    --num_classes=${num_classes} \
    --model_name=${model_name} \
    --use_decoder=True \
    --deploy_output_resizer=False

${TFBAZEL}/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph="${DEPLOY_DIR}/${frozen_name}.pb" \
    --out_graph="${DEPLOY_DIR}/${deploy_name}.pb" \
    --inputs=${input_node} \
    --outputs=${output_node} \
    --transforms='strip_unused_nodes
                  remove_nodes(op=Identity, op=CheckNumerics)
                  fold_constants(ignore_errors=true)
                  fold_batch_norms
                  fold_old_batch_norms'

python ${ROOT}/my_graph_utils/pb2pbtxt.py \
    --pb_path="${DEPLOY_DIR}/${deploy_name}.pb" \
    --log_dir=${DEPLOY_DIR} \
    --pbtxt_name="${deploy_name}.pbtxt"

python ${ROOT}/tf_coreml_utils/tf2coreml.py \
    --input_pb_file="${DEPLOY_DIR}/${deploy_name}.pb" \
    --output_mlmodel="${DEPLOY_DIR}/${deploy_name}.mlmodel" \
    --input_node_name=${input_node} \
    --output_node_name=${output_node}


