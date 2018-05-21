#!/bin/bash


#for name in "train" "validation" "test"
for name in "train"
do
    img="/home/xuyithu/openimages/big_person_and_face/${name}"
#    img="gs://xuyi-img/openimages/${name}"
    anno="/home/xuyithu/openimages/big_person_and_face/${name}-annotations-bbox.csv"
    #labelmap="/home/xuyithu/oid/class-descriptions-boxable.csv"
    labelmap="/home/xuyithu/workspace/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt"
    tfrecord="/home/xuyithu/openimages/big_person_and_face/tfrecord/${name}"
    num_shards=10

#    gsutil -m rsync -r gs://xuyi-img/openimages/${name} ${img}
    
    python object_detection/dataset_tools/create_oid_tf_record.py \
      --input_images_directory=${img} \
      --input_annotations_csv=${anno} \
      --input_label_map=${labelmap} \
      --output_tf_record_path_prefix=${tfrecord} \
      --num_shards=${num_shards}

done


