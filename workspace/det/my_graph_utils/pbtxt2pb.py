import tensorflow as tf
from google.protobuf import text_format



tf.app.flags.DEFINE_string(
    'pbtxt_path',
    'reshape_removed.pbtxt',
    'Path to the input pbtxt file')

tf.app.flags.DEFINE_string(
    'output_dir',
    './',
    'Directory to the output pb file')

tf.app.flags.DEFINE_string(
    'pb_name',
    'reshape_removed.pb',
    'Name of the output pb file')

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
  with open(FLAGS.pbtxt_path) as f:
    txt = f.read()
  gdef = text_format.Parse(txt, tf.GraphDef())
  tf.train.write_graph(gdef, FLAGS.output_dir, FLAGS.pb_name, as_text=False)

if __name__ == '__main__':
  tf.app.run()


