#!/bin/bash

HOME="/home/corp.owlii.com/yi.xu"
TFBAZEL="${HOME}/tensorflow/tensorflow/bazel-bin"
ROOT="${HOME}/workspace/models"
TFDET="${ROOT}/research/object_detection"
WORKSPACE="${ROOT}/workspace/det/pretrained"
PIPELINE_CONFIG="${WORKSPACE}/pipeline.config"
TRAIN_DIR="${WORKSPACE}/train"

CKPT="${WORKSPACE}/coco_ckpt/model.ckpt-34205"
DEPLOY="${TRAIN_DIR}/deploy"
mkdir -p ${DEPLOY}

deploy_graph_name="deploy_graph"
deploy_input_nodes="Preprocessor/sub:0"
deploy_output_nodes="box_encodings:0,class_indices:0,score_sigmoids:0"
ml_input_nodes=${deploy_input_nodes}
#ml_output_nodes=${deploy_output_nodes}
ml_output_nodes="all"

###### deploy for mobile app, no pre- or post-processing ######
echo "Exporting CoreML model..."
pretransform_graph_name="pretransform_graph"
python ${TFDET}/export_inference_graph.py \
    --input_type="image_tensor" \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix=${CKPT} \
    --output_directory=${DEPLOY} \
    --output_graph_name="${pretransform_graph_name}.pb" \
    --deploy=True

${TFBAZEL}/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph="${DEPLOY}/${pretransform_graph_name}.pb" \
    --out_graph="${DEPLOY}/${deploy_graph_name}.pb" \
    --inputs=${deploy_input_nodes} \
    --outputs=${deploy_output_nodes} \
    --transforms='strip_unused_nodes(type=float, shape="1,300,300,3")
                  remove_nodes(op=Identity, op=CheckNumerics)
                  fold_constants(ignore_errors=true)
                  fold_batch_norms
                  fold_old_batch_norms'

python ${WORKSPACE}/../my_graph_utils/pb2pbtxt.py \
    --pb_path="${DEPLOY}/${deploy_graph_name}.pb" \
    --log_dir=${DEPLOY} \
    --pbtxt_name="${deploy_graph_name}.pbtxt"

python ${WORKSPACE}/../tf_coreml_utils/tf2coreml.py \
    --input_pb_file="${DEPLOY}/${deploy_graph_name}.pb" \
    --output_mlmodel="${DEPLOY}/${deploy_graph_name}.mlmodel" \
    --input_node_name=${ml_input_nodes} \
    --output_node_name=${ml_output_nodes}

echo ""
echo "Done with exporting CoreML model!!!"
echo ""
###############################################################


##################################################
###### export graph without post-processing ######
echo "Exporting prepost graph..."
#tf_input_node="image_tensor:0"
#output_node="box_encodings:0,class_indices:0,score_sigmoids:0,anchors:0"
prepost_graph_name="prepost_graph"
PREPOST_DEPLOY="${DEPLOY}/prepost"

python ${TFDET}/export_inference_graph.py \
    --input_type="image_tensor" \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix=${CKPT} \
    --output_directory=${PREPOST_DEPLOY} \
    --output_graph_name="${prepost_graph_name}.pb" \
    --deploy=True \
    --output_anchors=True

python ${WORKSPACE}/../my_graph_utils/pb2pbtxt.py \
    --pb_path="${PREPOST_DEPLOY}/${prepost_graph_name}.pb" \
    --log_dir=${PREPOST_DEPLOY} \
    --pbtxt_name="${prepost_graph_name}.pbtxt"

echo ""
echo "Done with exporting prepost model!!!"
echo ""
##################################################


######################################
###### export for tf inference #######
echo "Exporting tf graph for inference..."
#tf_input_node="image_tensor:0"
#tfoutput_node="detection_boxes:0,detection_scores:0,detection_classes:0,num_detections:0"
tf_graph_name="tf_inference_graph"
TF_DEPLOY="${DEPLOY}/tf_inference"

python ${TFDET}/export_inference_graph.py \
    --input_type="image_tensor" \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix=${CKPT} \
    --output_directory=${TF_DEPLOY} \
    --output_graph_name="${tf_graph_name}.pb"

python ${WORKSPACE}/../my_graph_utils/pb2pbtxt.py \
    --pb_path="${TF_DEPLOY}/${tf_graph_name}.pb" \
    --log_dir=${TF_DEPLOY} \
    --pbtxt_name="${tf_graph_name}.pbtxt"

echo ""
echo "Done with exporting tf inference model!!!"
echo ""
#######################################



