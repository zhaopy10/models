#!/bin/bash


DATA="/home/corp.owlii.com/yi.xu/data/coco2017"
ANNO="${DATA}/annotations"
TRAIN_IMAGE_DIR="${DATA}/images"
VAL_IMAGE_DIR="${DATA}/val2017"
TEST_IMAGE_DIR="${DATA}/test2017"
TRAIN_ANNOTATIONS_FILE="${ANNO}/instances_train2017.json"
VAL_ANNOTATIONS_FILE="${ANNO}/instances_val2017.json"
TESTDEV_ANNOTATIONS_FILE="${ANNO}/instances_test2017.json"
OUTPUT_DIR="${DATA}/detection/person"

python object_detection/dataset_tools/create_coco_tf_record.py \
    --logtostderr \
    --train_image_dir="${TRAIN_IMAGE_DIR}" \
    --val_image_dir="${VAL_IMAGE_DIR}" \
    --test_image_dir="${TEST_IMAGE_DIR}" \
    --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
    --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
    --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
    --output_dir="${OUTPUT_DIR}" \
    --just_person=True


