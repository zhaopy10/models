import tensorflow as tf
import tfcoreml as tf_converter

tf.app.flags.DEFINE_string(
    'input_pb_file',
    '/home/corp.owlii.com/yi.xu/workspace/sgmt/train/deploy/deploy_graph.pb',
    'Input tensorflow pb file')

tf.app.flags.DEFINE_string(
    'output_mlmodel',
    '/home/corp.owlii.com/yi.xu/workspace/sgmt/train/deploy/deploy_graph.mlmodel',
    'Output coreml model file')

tf.app.flags.DEFINE_string(
    'input_node_name',
    'image:0',
    'As mentioned')

tf.app.flags.DEFINE_string(
    'output_node_name',
    'MobilenetV2/ResizeBilinear:0',
    'As mentioned')

tf.app.flags.DEFINE_multi_integer(
    'input_size', 
    [1, 512, 512, 3], 
    'As mentioned')

FLAGS = tf.app.flags.FLAGS

output_feature_names = ['MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd:0', FLAGS.output_node_name]
if FLAGS.output_node_name == 'None':
  output_feature_names = ['MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd:0']
def main(unused_argv):
  tf_converter.convert(tf_model_path = FLAGS.input_pb_file,
                       mlmodel_path = FLAGS.output_mlmodel,
                       output_feature_names = output_feature_names,
                       image_input_names = [FLAGS.input_node_name],
                       input_name_shape_dict = {FLAGS.input_node_name : FLAGS.input_size})
#                       red_bias = -123.68,
#                       green_bias = -116.78, 
#                       blue_bias = -103.94)
#                       image_scale = 2.0/255.0)


if __name__ == '__main__':
  tf.app.run()


