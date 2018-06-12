import tensorflow as tf
import coremltools

tf.app.flags.DEFINE_string(
    'input_mlmodel', None, 'As mentioned')

tf.app.flags.DEFINE_string(
    'output_mlmodel', None, 'As mentioned')

tf.app.flags.DEFINE_string(
    'output_node', 'heatmap__0', 'As mentioned')

tf.app.flags.DEFINE_bool(
    'is_bgr', False, 'As mentioned')

FLAGS = tf.app.flags.FLAGS


def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):  
    """  
    Convert an output multiarray to be represented as an image  
    This will modify the Model_pb spec passed in.  
    Example:  
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel')  
        spec = model.get_spec()  
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False)  
        newModel = coremltools.models.MLModel(spec)  
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel')  
    Parameters  
    ----------  
    spec: Model_pb  
        The specification containing the output feature to convert  
    feature_name: str  
        The name of the multiarray output feature you want to convert  
    is_bgr: boolean  
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR  
    """  
    for output in spec.description.output:  
        if output.name != feature_name:  
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':  
            raise ValueError("%s is not a multiarray type" % output.name)  
        print('')
        print('Turning output node ' + output.name + ' of type MultiArray into Image (i.e. PixelBuffer in iOS.')
        array_shape = tuple(output.type.multiArrayType.shape)  
        channels, height, width = array_shape  
        from coremltools.proto import FeatureTypes_pb2 as ft  
        if channels == 1:  
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')  
        elif channels == 3:  
            if is_bgr:  
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')  
            else:  
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')  
        else:  
            raise ValueError("Channel Value %d not supported for image inputs" % channels)  
        output.type.imageType.width = width  
        output.type.imageType.height = height 


def main(unused_argv):
  model = coremltools.models.MLModel(FLAGS.input_mlmodel)
  spec = model.get_spec()
  convert_multiarray_output_to_image(spec, FLAGS.output_node, is_bgr=FLAGS.is_bgr)
  newModel = coremltools.models.MLModel(spec)
  newModel.save(FLAGS.output_mlmodel)
  print('Converted. Saved at ' + FLAGS.output_mlmodel + '.')

if __name__ == '__main__':
  tf.app.run()


