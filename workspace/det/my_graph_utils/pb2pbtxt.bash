#!/bin/bash

HOME="/home/corp.owlii.com/yi.xu/workspace/sgmt"

pb_path='/home/corp.owlii.com/yi.xu/workspace/sgmt/decoder/train/deploy/frozen_graph.pb'
log_dir='/home/corp.owlii.com/yi.xu/workspace/sgmt/decoder/train/deploy/'
pbtxt_name='frozen_graph.pbtxt'

python ${HOME}/my_graph_utils/pb2pbtxt.py \
    --pb_path=${pb_path} \
    --log_dir=${log_dir} \
    --pbtxt_name=${pbtxt_name}


