import os
from io import BytesIO
import numpy as np
from PIL import Image

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

INPUT_TENSOR_NAME = 'image:0'
OUTPUT_TENSOR_NAME = 'MobilenetV2/Logits/output:0'
INPUT_SIZE = 512
FROZEN_GRAPH_NAME = './temp/deploy_graph.pb'
#FROZEN_GRAPH_NAME = './train/deploy/deployed_graph/deploy_graph.pb'

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
    the_input = [np.asarray(resized_image)]


    t1 = time.time()
    batch_seg_map = self.sess.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: the_input})
    t2 = time.time()
    print('First time: ', (t2-t1)*1000)

    n = 50
    total = 0
    for i in range(n):
      t1 = time.time()
      batch_seg_map = self.sess.run(
          OUTPUT_TENSOR_NAME,
          feed_dict={INPUT_TENSOR_NAME: the_input})
      t2 = time.time()
      total = total + (t2 - t1)*1000
    total = total / n

    logits = batch_seg_map[0]
    label = np.argmax(logits, axis=2)
    return resized_image, logits, label, total


model = SgmtModel()
image_name = '0002_00000001.png'
img = Image.open(image_name)
embed = img.load()

resized_image, logits, label, total = model.run(img)

for i in range(16):
  for j in range(16):
    if label[j, i] == 1:
      label[j, i] = 255
#      r, g, b = embed[i, j]
#      embed[i, j] = (r+100, g, b+100)
#    else:
#      r, g, b = embed[i, j]
#      embed[i, j] = (r, g/2, b/2)

mask = Image.fromarray(np.uint8(label))
mask.save('mask.png')
#img.save('embed.png')

print('Total time: ', total)


