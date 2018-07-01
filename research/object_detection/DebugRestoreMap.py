import functools
import json
import os
import tensorflow as tf

from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy

slim = tf.contrib.slim

configs = config_util.get_configs_from_pipeline_file(
        'faster_rcnn_inception_v2_coco_debug.config')

model_config = configs['model']

model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)


train_config = configs['train_config']
print train_config

if train_config.from_detection_checkpoint:
    train_config.fine_tune_checkpoint_type = 'detection'
else:
    train_config.fine_tune_checkpoint_type = 'classification'

detection_model = model_fn()
print detection_model
print detection_model._number_of_stages

var_map = detection_model.restore_map()
'''
var_map = detection_model.restore_map(
          fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
          load_all_detection_checkpoint_vars=(
              train_config.load_all_detection_checkpoint_vars))
'''
print var_map
