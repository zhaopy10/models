#!/bin/bash

num_clones=3        # 3
batch_size=512      # 512
num_steps=250000    # 250000 steps, about 295 epochs

start_lr=0.045
end_lr=0.0001
factor=0.98

SLIM="/home/popzq/workspace/models/research/slim"
WORKSPACE="/home/popzq/workspace/models/workspace/cls"
DATASET_DIR="/home/popzq/disk/ILSVRC2012/train"
TRAIN_DIR="${WORKSPACE}/train"

python ${SLIM}/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --max_number_of_steps=${num_steps} \
    --model_name=mobilenet_v2_020 \
    --batch_size=${batch_size} \
    --num_clones=${num_clones} \
    --train_image_size=224 \
    --learning_rate_decay_type=exponential \
    --learning_rate_decay_factor=${factor} \
    --learning_rate=${start_lr} \
    --end_learning_rate=${end_lr} \
    --num_epochs_per_decay=1.0 \
    --dataset_name=imagenet \
    --num_readers=9 \
    --num_preprocessing_threads=9 \
    --log_every_n_steps=1 \
    --save_summaries_secs=600 \
    --save_interval_secs=600 \
    --max_ckpts_to_keep=3 \
    --keep_ckpt_every_n_hours=6.0



