#!/bin/bash

model_name="mobilenet_v2_035_sgmt"
log_dir="dec1/voc"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_name="pascal_voc_saliency"
tfrecord_dir="pascal_voc_saliency/tfrecord"

ckpt="dec1/ckpt/voc/model.ckpt-12"
num_clones=1
batch_size=16      # 192 * 2  
train_steps=12    # 6000 steps, about 31 epoches
lr_decay_factor=0.8

###########################################
HOME="/home/corp.owlii.com/yi.xu"
SLIM="${HOME}/workspace/models/research/slim"
WORKSPACE="${HOME}/workspace/models/workspace/seg"
DATASET_DIR="${HOME}/data/${tfrecord_dir}"
INIT_CHECKPOINT="${WORKSPACE}/${ckpt}"
TRAIN_DIR="${WORKSPACE}/${log_dir}"
mkdir -p ${TRAIN_DIR}

##### Start tuning #####
python train_sgmt.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${INIT_CHECKPOINT} \
  --max_number_of_steps=${train_steps} \
  --batch_size=${batch_size} \
  --min_scale_factor=0.5 \
  --max_scale_factor=2.0 \
  --scale_factor_step_size=0 \
  --learning_rate=0.000 \
  --learning_rate_decay_type=exponential \
  --learning_rate_decay_factor=${lr_decay_factor} \
  --save_interval_secs=300 \
  --save_summaries_secs=300 \
  --max_ckpts_to_keep=3 \
  --keep_ckpt_every_n_hours=2.0 \
  --log_every_n_steps=5 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --num_clones=${num_clones} \
  --use_decoder=True

# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${second_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=val \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop



