#!/bin/bash

#name=$1

for name in "train" "validation" "test"
do
    img="gs://xuyi-img/openimages/${name}"
    anno="/home/xuyithu/oid/${name}-annotations-bbox.csv"
    #labelmap="/home/xuyithu/oid/class-descriptions-boxable.csv"
    labelmap="./object_detection/data/oid_bbox_trainable_label_map.pbtxt"
    tfrecord="gs://xuyi-img/openimages/oid-tfrecord/${name}/${name}"
    
    python object_detection/dataset_tools/create_oid_tf_record.py \
      --input_images_directory=${img} \
      --input_annotations_csv=${anno} \
      --input_label_map=${labelmap} \
      --output_tf_record_path_prefix=${tfrecord}

done


