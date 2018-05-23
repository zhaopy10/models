import tensorflow as tf
from tensorflow.python.platform import gfile



tf.app.flags.DEFINE_string(
    'pb_path', 
    './deploy_graph.pb', 
    'Path to the input pb file')

tf.app.flags.DEFINE_string(
    'log_dir',
    './',
    'Directory to the output pbtxt file')

tf.app.flags.DEFINE_string(
    'pbtxt_name', 
    'deploy_graph.pbtxt',
    'Name of the output pbtxt file')

FLAGS = tf.app.flags.FLAGS


def converter(in_pb, log_dir, pbtxt_name): 
  with gfile.FastGFile(in_pb,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, log_dir, pbtxt_name, as_text=True)
  return



def main(unused_argv):
  converter(in_pb=FLAGS.pb_path, 
            log_dir=FLAGS.log_dir, 
            pbtxt_name=FLAGS.pbtxt_name)

if __name__ == '__main__':
  tf.app.run()





