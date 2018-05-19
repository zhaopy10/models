#!/bin/bash

# From tensorflow/models/research/
pip install --upgrade protobuf pillow numpy lxml
protoc object_detection/protos/*.proto --python_out=.
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

