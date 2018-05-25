#!/bin/bash

# From tensorflow/models/research/
pip install --upgrade protobuf pillow numpy lxml contextlib2 pandas
conda install protobuf
protoc object_detection/protos/*.proto --python_out=.
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

