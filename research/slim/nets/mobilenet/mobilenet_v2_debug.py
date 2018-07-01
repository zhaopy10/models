from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2

slim = tf.contrib.slim

def find_ops(optype):
  """Find ops of a given type in graphdef or a graph.

  Args:
    optype: operation type (e.g. Conv2D)
  Returns:
     List of operations.
  """
  gd = tf.get_default_graph()
  return [var for var in gd.get_operations() if var.type == optype]

if __name__ == '__main__':
  spec = dict(mobilenet_v2.V2_DEF)
  _, ep = mobilenet_v2.mobilenet(
    tf.placeholder(tf.float32, (10, 224, 224, 16)), conv_defs=spec,
    base_only=True)
  