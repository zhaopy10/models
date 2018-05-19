import tensorflow as tf
slim = tf.contrib.slim

def get_network_fn():
  def network_fn(images):
    return func(images)
  return network_fn


def func(images):
  net = images

  net = slim.avg_pool2d(net,
                        [1, 1],
                        stride=1,
                        padding='VALID',
                        scope='Pool2d')

  net = slim.separable_conv2d(net, 1, [1, 1], 1,
                              normalizer_fn=slim.batch_norm, 
                              weights_initializer=tf.constant_initializer(1),
                              biases_initializer=tf.zeros_initializer(),
                              trainable=True,
                              scope='Sepa2d')

  net = slim.conv2d(net, 2, [1,1], 
                    activation_fn=tf.nn.relu, 
                    normalizer_fn=slim.batch_norm, 
                    weights_initializer=tf.constant_initializer(0.333),
                    biases_initializer=tf.zeros_initializer(),
                    trainable=True,
                    scope='Conv2d')

  logits = tf.identity(net, name='output')

  return logits

