import os
import sys
from io import BytesIO
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import time


workspace = '/Users/yixu/workspace/coreml/'

model_dir = workspace + 'det/'
res_dir = model_dir + 'res/'
if not os.path.exists(res_dir):
  os.makedirs(res_dir)

data_dir = workspace + 'data/'
listpath = data_dir + 'det.txt'

FROZEN_GRAPH_NAME = model_dir + 'prepost_graph.pb'
INPUT_SIZE = 300

class SgmtModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_NAME = 'image_tensor:0'
  BOX_ENCODINGS = 'box_encodings:0'
  CLASS_SCORES = 'class_scores:0'
  ANCHORS = 'anchors:0'

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

    image = image.resize([300, 300], Image.BILINEAR)
    image.save('det.jpg')
    the_input = [np.asarray(image)]
    print('input_shape: ', the_input[0].shape)

    t1 = time.time()
    res = self.sess.run(
        {'box_encodings': self.BOX_ENCODINGS, 
         'class_scores': self.CLASS_SCORES, #},
         'anchors': self.ANCHORS},
        feed_dict={self.INPUT_NAME: the_input})
    t2 = time.time()
    total = (t2 - t1) * 1000

    return res, total


def infer_one(image, use_heatmap=True):
  # image preprocessing
#  image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.ANTIALIAS)
#  image.save('det.jpg')
  
  # run model
  model = SgmtModel()
  res, running_time = model.run(image)

  box_encodings = res['box_encodings']
  class_scores = res['class_scores']
  anchors = res['anchors']
#  anchors = None

  return box_encodings, class_scores, anchors, running_time



# now start inferring
with open(listpath) as f:
    lines = f.readlines()
lines = [x.strip('\n') for x in lines] 
#print(lines)

#for filename in lines:

filename = lines[0]
filename_root = os.path.splitext(filename)[0]

image = Image.open(data_dir + filename)
box_encodings, class_scores, anchors, running_time = infer_one(image)


#scores = class_scores[0,:,0,1]
#sorted_scores = np.sort(scores)
#print(sorted_scores[:10])
#print(sorted_scores[1888:])

#print(anchors[0,:])
#print(anchors[1,:])
#print(anchors[2,:])
#print(anchors[3,:])
#print(anchors[4,:])
#print(anchors[5,:])
#print(anchors[6,:])
#print(anchors[7,:])
#print(anchors[8,:])
#print(anchors[9,:])
#print(anchors[1916,:])


sorted_scores = np.sort(class_scores[0,:,0,1])
print('Top 20 person scores: ', sorted_scores[class_scores.shape[1] - 20:])

anchor_shape = anchors.shape
print('anchors', anchor_shape)
print('box_encodings', box_encodings.shape)
print('class_scores', class_scores.shape)

# write results to file
from array import array
output_file = open(model_dir + 'anchors.bin', 'wb')
for i in range(anchor_shape[0]):
  float_array = array('f', anchors[i,:])
  float_array.tofile(output_file)
output_file.close()

output_file = open(model_dir + 'box_encodings.bin', 'wb')
for i in range(box_encodings.shape[1]):
  float_array = array('f', box_encodings[0,i,0,:])
  float_array.tofile(output_file)
output_file.close()

output_file = open(model_dir + 'class_scores.bin', 'wb')
#float_array = array('f', [class_scores.shape[3]])
#float_array.tofile(output_file)
print(class_scores.shape[3])
for i in range(class_scores.shape[3]):
  float_array = array('f', class_scores[0,:,0,i])
  float_array.tofile(output_file)
output_file.close()




