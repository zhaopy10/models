#!/bin/bash

# From tensorflow/models/research/
cd ~/workspace/models/research

# install dependencies
pip install --upgrade protobuf pillow numpy lxml contextlib2 pandas
conda install protobuf
protoc object_detection/protos/*.proto --python_out=.

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../research/

# compile proto
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


