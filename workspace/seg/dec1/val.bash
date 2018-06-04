#!/bin/bash


log_dir="dec1/voc"
model_name="mobilenet_v2_035_sgmt"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_name="pascal_voc_saliency"
tfrecord_dir="pascal_voc_saliency/tfrecord"

###################################
HOME="/home/popzq"
SLIM="${HOME}/workspace/models/research/slim"
WORKSPACE="${HOME}/workspace/models/workspace/seg"
DATASET_DIR="${HOME}/disk/${tfrecord_dir}"
VAL_DIR="${WORKSPACE}/${log_dir}"

# Run evaluation.
python eval_sgmt.py \
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


