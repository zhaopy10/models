import os
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf

INPUT_SIZE = 224

INPUT_TENSOR_NAME = 'input:0'
OUTPUT_TENSOR_NAME = 'MobilenetV2/Predictions/Reshape_1:0'
FROZEN_GRAPH_NAME = './mobilenet_v2_1.0_224_frozen.pb'

#INPUT_TENSOR_NAME = 'MobilenetV2/input:0'
#OUTPUT_TENSOR_NAME = 'MobilenetV2/Conv/Relu6:0'
#OUTPUT_TENSOR_NAME = 'MobilenetV2/Conv/BatchNorm/FusedBatchNorm:0'
#OUTPUT_TENSOR_NAME = 'MobilenetV2/Predictions/Reshape_1:0'
#FROZEN_GRAPH_NAME = './frozen_graph.pb'


class SgmtModel(object):
  """Class to load deeplab model and run inference."""

#  INPUT_TENSOR_NAME = 'MobilenetV2/input:0'
#  OUTPUT_TENSOR_NAME = 'ResizeBilinear_1:0'
#  INPUT_SIZE = 512
#  FROZEN_GRAPH_NAME = './train.035.1280/deploy/frozen_graph.pb'

  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    with tf.gfile.GFile(FROZEN_GRAPH_NAME, "rb") as f:
      print(FROZEN_GRAPH_NAME)
      graph_def = tf.GraphDef().FromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    batch_seg_map = self.sess.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    logits = batch_seg_map[0]
    label = np.argmax(logits, axis=0)

 #   print(batch_seg_map.shape)

#    return batch_seg_map
    return resized_image, logits, label


model = SgmtModel()
image_name = './flowers/rose/10503217854_e66a804309.jpg'
img = Image.open(image_name)
img = img.resize([INPUT_SIZE,INPUT_SIZE], Image.ANTIALIAS)

resized_image, logits, label = model.run(img)
print('Max prob label: ', label)
for i in range(5):
    pred = np.argmax(logits, axis=0)
    prob = logits[pred]
    logits[pred] = 0
    print(pred, prob)

#logits = model.run(img)
#c = 10
#feat = np.ones((112,112))
#for i in range(112):
#  for j in range(112):
#    feat[i,j] = logits[0,i,j,10] * 40
#print(feat[0,:])
#feat = Image.fromarray(np.uint8(feat))
#feat.save('feat_tf.png')


