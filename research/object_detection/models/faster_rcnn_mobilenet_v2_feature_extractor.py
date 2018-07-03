# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Mobilenet v1 Faster R-CNN implementation."""
import numpy as np

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import shape_utils
#from nets import mobilenet_v1
from nets.mobilenet import mobilenet_v2 as mobilenet_v2
from nets.mobilenet import mobilenet as lib
op = lib.op

from nets.mobilenet import conv_blocks as ops

slim = tf.contrib.slim

'''
def _get_mobilenet_conv_no_last_stride_defs(conv_depth_ratio_in_percentage):
#  print('###models/faster_rcnn_mobilenet_v2_feature_extractor.py###')
# This func is not called because self._skip_last_stride is not true
  return [
      op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
      op(ops.expanded_conv,
         expansion_size=expand_input(1, divisible_by=1),
         num_outputs=16),
      op(ops.expanded_conv, stride=2, num_outputs=24),
      op(ops.expanded_conv, stride=1, num_outputs=24),
      op(ops.expanded_conv, stride=2, num_outputs=32),
      op(ops.expanded_conv, stride=1, num_outputs=32),
      op(ops.expanded_conv, stride=1, num_outputs=32),
      op(ops.expanded_conv, stride=2, num_outputs=64),
      op(ops.expanded_conv, stride=1, num_outputs=64),
      op(ops.expanded_conv, stride=1, num_outputs=64),
      op(ops.expanded_conv, stride=1, num_outputs=64),
      op(ops.expanded_conv, stride=1, num_outputs=96),
      op(ops.expanded_conv, stride=1, num_outputs=96),
      op(ops.expanded_conv, stride=1, num_outputs=96),
      op(ops.expanded_conv, stride=2, num_outputs=160),
      op(ops.expanded_conv, stride=1, num_outputs=160),
      op(ops.expanded_conv, stride=1, num_outputs=160),
      op(ops.expanded_conv, stride=1, num_outputs=320),
      op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
  ]
'''


class FasterRCNNMobilenetV2FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Mobilenet V2 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               depth_multiplier=1.0,
               min_depth=16,
               skip_last_stride=False,
               conv_depth_ratio_in_percentage=100):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      skip_last_stride: Skip the last stride if True.
      conv_depth_ratio_in_percentage: Conv depth ratio in percentage. Only
        applied if skip_last_stride is True.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._skip_last_stride = skip_last_stride
    self._conv_depth_ratio_in_percentage = conv_depth_ratio_in_percentage
    super(FasterRCNNMobilenetV2FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

#    print('###faster_rcnn_mobilenet_v2_feature_extractor.py### - depth_multiplier: ',
#        depth_multiplier)


  def preprocess(self, resized_inputs):
    """Faster R-CNN Mobilenet V2 preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """

#    print('###faster_rcnn_mobilenet_v2_feature_extractor.py### - extract_proposal_features')

    preprocessed_inputs.get_shape().assert_has_rank(4)
    preprocessed_inputs = shape_utils.check_min_image_dim(
        min_dim=33, image_tensor=preprocessed_inputs)

    with slim.arg_scope(
        mobilenet_v2.training_scope(
            is_training=self._train_batch_norm,
            weight_decay=self._weight_decay)):
      with tf.variable_scope('MobilenetV2',
                             reuse=self._reuse_weights) as scope:
        params = {}
        '''
        if self._skip_last_stride:
          # Not called by default, will use conv_defs in slim.nets.mobilenet.mobilenet_v2
          params['conv_defs'] = _get_mobilenet_conv_no_last_stride_defs(
              conv_depth_ratio_in_percentage=self.
              _conv_depth_ratio_in_percentage)
        '''
        '''
        # add by yi.xu
        _, endpoints = mobilenet_v2.mobilenet_base(
            preprocessed_inputs,
            final_endpoint='layer_19',  # actually 'MobilenetV2/Conv_1'
            min_depth=self._min_depth,
            depth_multiplier=self._depth_multiplier,
            scope=scope,
            **params)
        '''
        # pyz add: test v2
        _, endpoints = mobilenet_v2.mobilenet(
            preprocessed_inputs,
            final_endpoint='layer_19',  # layer_14 is the last layer, use layer_18 instead
            base_only=True,
            min_depth=self._min_depth,
            depth_multiplier=self._depth_multiplier,
            scope=scope,
            **params)
        # pyz comments: it seems that depth of layer_18 should be 320
        print 'layer_19 shape: ', endpoints['layer_19'].get_shape()
    return endpoints['layer_19'], endpoints

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    net = proposal_feature_maps

    conv_depth = 1024
    '''
    # ignore
    if self._skip_last_stride:
      conv_depth_ratio = float(self._conv_depth_ratio_in_percentage) / 100.0
      conv_depth = int(float(conv_depth) * conv_depth_ratio)
    '''

    depth = lambda d: max(int(d * 1.0), 16)
    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights):
      with slim.arg_scope(
          mobilenet_v2.training_scope(
              is_training=self._train_batch_norm,
              weight_decay=self._weight_decay)):

        # it is the last two layers of mobilenet v1, should be changed
        '''
        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d], padding='SAME'):

          net = slim.separable_conv2d(
              net,
              depth(1280), [3, 3],
              depth_multiplier=1,
              stride=2,
              scope='Conv_2')  # or 'layer_20'
          return slim.separable_conv2d(
              net,
              depth(1280), [3, 3],
              depth_multiplier=1,
              stride=1,
              scope='Conv_3')  # or 'layer_21'

        '''

        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d], padding='SAME'), \
            slim.arg_scope([slim.batch_norm], is_training=self._train_batch_norm):
            # not sure, but i think bn should be trainable
          net = slim.separable_conv2d(
              net, None, [3, 3],
              depth_multiplier=1,
              stride=2, rate=1, normalizer_fn=slim.batch_norm,
              scope='add_conv2d_1_depthwise')
          net = slim.conv2d(net, depth(conv_depth), [1, 1],
              stride=1,
              normalizer_fn=slim.batch_norm,
              scope='add_conv2d_1_pointwise')
          net = slim.separable_conv2d(
              net, None, [3, 3],
              depth_multiplier=1,
              stride=1, rate=1, normalizer_fn=slim.batch_norm,
              scope='add_conv2d_2_depthwise')
          net = slim.conv2d(net, depth(conv_depth), [1, 1],
              stride=1,
              normalizer_fn=slim.batch_norm,
              scope='add_conv2d_2_pointwise')
          return net
'''
          net = ops.expanded_conv(
              net,
              stride=2, num_outputs=depth(160),
              normalizer_fn=slim.batch_norm,
              normalizer_params={'scale': True},
              scope='expanded_conv_13'
              )
          net = ops.expanded_conv(
              net,
              stride=1, num_outputs=depth(160),
              normalizer_fn=slim.batch_norm,
              normalizer_params={'scale': True},
              scope='expanded_conv_14'
              )
          net = ops.expanded_conv(
              net,
              stride=1, num_outputs=depth(160),
              normalizer_fn=slim.batch_norm,
              normalizer_params={'scale': True},
              scope='expanded_conv_15'
              )
          net = ops.expanded_conv(
              net,
              stride=1, num_outputs=depth(320),
              normalizer_fn=slim.batch_norm,
              normalizer_params={'scale': True},
              scope='expanded_conv_16'
              )
          return slim.conv2d(
              net,
              num_outputs = depth(1280), kernel_size = [1, 1],
              stride=1, normalizer_fn=slim.batch_norm,
              normalizer_params={'scale': True},
              scope='Conv_1')
'''



