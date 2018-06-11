#!/bin/bash

data_dir="/home/xuyithu/workspace/oid.head.nogroupimages"
TFDET="/home/xuyithu/workspace/models/research/object_detection"

for name in "train" "validation"
do
    img="${data_dir}/${name}"
#    img="gs://xuyi-img/openimages/${name}"
    anno="${data_dir}/${name}-annotations-bbox.csv"
    labelmap="./oid_label_map.pbtxt"
    tfrecord="${data_dir}/tfrecord/${name}"
    mkdir -p "${data_dir}/tfrecord"

    num_shards=10

    python ${TFDET}/dataset_tools/create_oid_tf_record.py \
      --input_images_directory=${img} \
      --input_annotations_csv=${anno} \
      --input_label_map=${labelmap} \
      --output_tf_record_path_prefix=${tfrecord} \
      --num_shards=${num_shards}

done


