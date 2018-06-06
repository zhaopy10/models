#!/bin/bash

log_dir="person.instance/train"
model_name="mobilenet_v2_035_sgmt"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
#dataset_name="pascal_voc_saliency"
#tfrecord_dir="pascal_voc_saliency/tfrecord"
dataset_name="person_instance"
tfrecord_dir="coco2017/person_instance/tfrecord"

###################################
HOME="/home/corp.owlii.com/yi.xu"
SLIM="${HOME}/workspace/models/research/slim"
WORKSPACE="${HOME}/workspace/models/workspace/seg"
DATASET_DIR="${HOME}/data/${tfrecord_dir}"
VAL_DIR="${WORKSPACE}/${log_dir}"

# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${VAL_DIR} \
  --eval_dir=${VAL_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=val \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=1 \
  --use_cpu=True \
  --eval_interval_secs=10 \
  --use_decoder=True \
  --max_number_of_evaluations 0  # 0 for infinite loop


