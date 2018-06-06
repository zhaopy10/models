#!/bin/bash

##### This is for deploying a mlmodel with softmax layer, 
##### which requires mannually removing Reshape layer in pbtxt file 
##### and then convert it to mlmodel.

# Step 0, manually prepare a pbtxt file with Reshape layer removed.
# Then do the following.
#ori_loc="/OwlNAS/OwlDocuments/tmp/xy/temp/reshape_removed.pbtxt"
ori_loc="./reshape_removed.pbtxt"
log_dir="dec1/voc"
name="sfm_graph"
pbtxt_name="${name}.pbtxt"

WORKSPACE="/home/corp.owlii.com/yi.xu/workspace/models/workspace/seg"
DEPLOY_DIR="${WORKSPACE}/${log_dir}/sfm_ml"
mkdir -p ${DEPLOY_DIR}
cp "${ori_loc}" "${DEPLOY_DIR}/${pbtxt_name}"

pb_name="${name}.pb"

mlmodel="${name}.mlmodel"
input_node="image:0"
ml_output_node="MobilenetV2/heatmap:0"

#####################################
python ${WORKSPACE}/my_graph_utils/pbtxt2pb.py \
    --pbtxt_path="${DEPLOY_DIR}/${pbtxt_name}" \
    --output_dir="${DEPLOY_DIR}" \
    --pb_name="${pb_name}"


python ${WORKSPACE}/tf_coreml_utils/tf2coreml.py \
    --input_pb_file="${DEPLOY_DIR}/${pb_name}" \
    --output_mlmodel="${DEPLOY_DIR}/${mlmodel}" \
    --input_node_name=${input_node} \
    --output_node_name=${ml_output_node}


