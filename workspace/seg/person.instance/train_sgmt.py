"""
Training script adjusted from slim.train_image_classifier.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
#sys.path.append('/home/corp.owlii.com/yi.xu/tensorflow/models/research')
import deeplab
from deeplab import common
#from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from datasets import dataset_factory
from deployment import model_deploy
#from nets import nets_factory
from preprocessing import preprocessing_factory

sys.path.append('..')
from my_data_utils import segmentation_dataset

from local_net_fn import nets_factory

slim = tf.contrib.slim



tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'pascal_voc_seg', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2_sgmt', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

#tf.app.flags.DEFINE_integer(
#    'train_image_size', 224, 'Train image size')
tf.app.flags.DEFINE_multi_integer(
    'train_crop_size', [512, 512],
    'Image crop size [height, width] during training.')

tf.app.flags.DEFINE_multi_integer(
    'output_size', [512, 512], 'As mentioned')

#tf.app.flags.DEFINE_integer(
#    'min_resize_value', None, 'See min_resize_value in Deeplab.')

#tf.app.flags.DEFINE_integer(
#    'max_resize_value', None, 'See max_resize_value in Deeplab.')

#tf.app.flags.DEFINE_integer(
#    'resize_factor', None, 'As mentioned..')

tf.app.flags.DEFINE_float(
    'min_scale_factor', 0.5, 'See min_scale_factor in Deeplab.')

tf.app.flags.DEFINE_float(
    'max_scale_factor', 2.0, 'See max_scale_factor in Deeplab.')

tf.app.flags.DEFINE_float(
    'scale_factor_step_size', 0.0, 'See scale_factor_step_size in Deeplab.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

#####################
# Other flags #
#####################

tf.app.flags.DEFINE_integer(
    'max_ckpts_to_keep', 3, 'As mentioned')

tf.app.flags.DEFINE_float(
    'keep_ckpt_every_n_hours', 1.0, 'As mentioned')

tf.app.flags.DEFINE_boolean(
    'adjust_for_deploy', False, 'As mentioned')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
#    dataset = dataset_factory.get_dataset(
#        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, dataset_dir=FLAGS.dataset_dir)
    # dataset.num_classes = 21

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=(not FLAGS.adjust_for_deploy))

    #####################################
    # Select the preprocessing function #
    #####################################
#    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#        preprocessing_name,
#        is_training=(not FLAGS.adjust_for_deploy))

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
#      provider = slim.dataset_data_provider.DatasetDataProvider(
#          dataset,
#          num_readers=FLAGS.num_readers,
#          common_queue_capacity=20 * FLAGS.batch_size,
#          common_queue_min=10 * FLAGS.batch_size)
#      [image, label] = provider.get(['image', 'label'])
#      label -= FLAGS.labels_offset
#
#      train_image_size = FLAGS.train_image_size or network_fn.default_image_size
#      # train_image_size = 224
#
#      image = image_preprocessing_fn(image, train_image_size, train_image_size)
#
#      images, labels = tf.train.batch(
#          [image, label],
#          batch_size=FLAGS.batch_size,
#          num_threads=FLAGS.num_preprocessing_threads,
#          capacity=5 * FLAGS.batch_size)
#      labels = slim.one_hot_encoding(
#          labels, dataset.num_classes - FLAGS.labels_offset)
#      batch_queue = slim.prefetch_queue.prefetch_queue(
#          [images, labels], capacity=2 * deploy_config.num_clones)

      clone_batch_size = int(FLAGS.batch_size / deploy_config.num_clones)

      samples = input_generator.get(
          dataset,
          FLAGS.train_crop_size,
          clone_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          dataset_split=FLAGS.dataset_split_name,
          is_training=True,
          model_variant=FLAGS.model_variant)

      batch_queue = slim.prefetch_queue.prefetch_queue(
          samples, capacity=128 * deploy_config.num_clones)


    ####################
    # Define the model #
    ####################
#    def clone_fn(batch_queue):
#      """Allows data parallelism by creating multiple clones of network_fn."""
#      images, labels = batch_queue.dequeue()
#      logits, end_points = network_fn(images)
#      print('images.shape: ', images.shape)  # (32, 224, 224, 3)
#      print('labels.shape: ', labels.shape)  # (32, 5)
#      print('logits.shape: ', logits.shape)  # (32, 5)
#
#      #############################
#      # Specify the loss function #
#      #############################
#      # There is no AuxLogits for mobilenet
#      if 'AuxLogits' in end_points:
#        slim.losses.softmax_cross_entropy(
#            end_points['AuxLogits'], labels,
#            label_smoothing=FLAGS.label_smoothing, weights=0.4,
#            scope='aux_loss')
#      slim.losses.softmax_cross_entropy(
#          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
#      return end_points
    def clone_fn(batch_queue, num_classes, ignore_label):
      samples = batch_queue.dequeue()

      samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name = common.IMAGE)
      samples[common.LABEL] = tf.identity(samples[common.LABEL], name = common.LABEL)

      images = samples[common.IMAGE]
      labels = samples[common.LABEL]

      # make predictions!
      logits, end_points = network_fn(images)

      print('images.shape: ', images.shape)
      print('labels.shape: ', labels.shape)
      print('logits.shape: ', logits.shape)

      # define loss
      if FLAGS.adjust_for_deploy:
        logits = tf.image.resize_bilinear(
            logits, FLAGS.output_size, align_corners=True)
      else:
        logits = tf.image.resize_bilinear(
            logits, tf.shape(labels)[1:3], align_corners=True)
#        heatmap = tf.nn.softmax(logits)
#        heatmap = tf.slice(heatmap, [0, 0, 0, 1], [clone_batch_size,
#                           FLAGS.output_size[0], FLAGS.output_size[1], 1], name='heatmap')
        # heatmap = tf.squeeze(heatmap, name='heatmap')

      print('upsampled logits shape: ', logits.shape)

      scaled_labels = labels
      scaled_labels = tf.reshape(scaled_labels, shape=[-1])
      ignore_label = dataset.ignore_label
      loss_weight = 1.0
      not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                 ignore_label)) * loss_weight
      one_hot_labels = slim.one_hot_encoding(
          scaled_labels, num_classes, on_value=1.0, off_value=0.0)

      print('one_hot_labels.shape: ', one_hot_labels.shape)

      tf.losses.softmax_cross_entropy(
           one_hot_labels,
           tf.reshape(logits, shape=[-1, num_classes]),
           weights=not_ignore_mask,
           scope=None)  # or scope = loss_scope = 'semantic'? I don't know.

      return end_points



    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    model_args = (batch_queue, dataset.num_classes, dataset.ignore_label)
    clones = model_deploy.create_clones(deploy_config, clone_fn, args=model_args)
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Add config to avoid 'could not satisfy explicit device' problem 
    sess_config = tf.ConfigProto(allow_soft_placement=True)

    ###########################
    # Kicks off the training. #
    ###########################
    saver = tf.train.Saver(
                max_to_keep=FLAGS.max_ckpts_to_keep, 
                keep_checkpoint_every_n_hours=FLAGS.keep_ckpt_every_n_hours)

    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None,
        session_config=sess_config, 
        saver=saver)


if __name__ == '__main__':
  tf.app.run()