#!/bin/bash


project="035.dec1.3x4"
log_dir="train"
model_name="mobilenet_v2_035_sgmt"

###### data ######
#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_name="pascal_voc_saliency"
tfrecord_dir="pascal_voc_saliency/tfrecord"

###### directories ######
#HOME="/home/corp.owlii.com/yi.xu"
ROOT="${HOME}/workspace/models/workspace/seg"
WORKSPACE="${ROOT}/${project}"
TRAIN_UTILS="${ROOT}/my_training_utils"
DATASET_DIR="${HOME}/data/${tfrecord_dir}"
VAL_DIR="${WORKSPACE}/${log_dir}"
PYTHONPATH="${PYTHONPATH}:${WORKSPACE}:${WORKSPACE}/.."

# Run evaluation.
python ${TRAIN_UTILS}/eval_sgmt.py \
  --checkpoint_path=${VAL_DIR} \
  --eval_dir=${VAL_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=val \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=16 \
  --use_cpu=False \
  --eval_interval_secs=10 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --use_decoder=True \
  --max_number_of_evaluations 0  # 0 for infinite loop


