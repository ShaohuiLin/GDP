# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
import pdb

from datasets import dataset_factory
from deployment import model_deploy
from keras_nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', 'tmp/tfmodel/',
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
    'log_every_n_steps', 100,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 6000,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_max_keep', 5,
    'The most model to save')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
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

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 10,
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
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'tfrecord', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'vgg_16', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

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
# GDP Flags #
#####################
tf.app.flags.DEFINE_integer(
    'update_mask_step', 500,
    'per step to update mask')

tf.app.flags.DEFINE_float(
    'beta', 0.5,
    '1 - compression ratio')

tf.app.flags.DEFINE_integer(
    'taylor_step', 100,
    'the number of batch size to calculate taylor value')

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


def make_keymap(checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path)
    reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keymap = dict()
    kvmap = dict()
    for key in var_to_shape_map:

        if key == "global_step" or "mean_rgb" in key or "AuxLogits" in key:
            continue

        newkey = key
        if "resnet" in key and "weights" in key and ("shortcut" in key or "conv" in key):
            newkey = key.replace("weights", "conv2d/kernel")
        elif "Inception" in key and "weight" in key and "Logits" not in key:
            newkey = key.replace("weights", "conv2d/kernel")
        elif "weights" in key:
            newkey = key.replace("weights", "kernel")
        elif "biases" in key:
            newkey = key.replace("biases", "bias")
        model_name = key.split("/")[0]
        if FLAGS.model_name != model_name:
            newkey = newkey.replace(model_name, FLAGS.model_name)

        keymap[key] = newkey
        kvmap[newkey] = reader.get_tensor(key)

    return keymap, kvmap


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
    An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

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

    keymap, kvmap = make_keymap(checkpoint_path)
    var_names_to_values = dict()
    for var in tf.global_variables():
        if var.name[:-2] in kvmap.keys():
            var_names_to_values[var.name[:-2]] = kvmap[var.name[:-2]]

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_values_fn(var_names_to_values)



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


def train_step(sess, train_op, global_step, train_step_kwargs):
    start_time = time.time()

    trace_run_options = None
    run_metadata = None
    if 'should_trace' in train_step_kwargs:
        if 'logdir' not in train_step_kwargs:
          raise ValueError('logdir must be present in train_step_kwargs when '
                           'should_trace is present')
        if sess.run(train_step_kwargs['should_trace']):
          trace_run_options = config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE)
          run_metadata = config_pb2.RunMetadata()

    np_global_step = sess.run(global_step)

    if np_global_step % FLAGS.update_mask_step == 0:
        K.set_learning_phase(False)

        gdp_mask_taylor = train_step_kwargs['gdp_mask_taylor']
        network_filters = []
        mean_tvs = [np.zeros(g[1].shape[0]) for g in gdp_mask_taylor]

        for i in range(len(gdp_mask_taylor)):
            mask_index = sess.graph.get_tensor_by_name(gdp_mask_taylor[i][0].name.replace('mask', 'mask_index'))
            mask_value = sess.graph.get_tensor_by_name(gdp_mask_taylor[i][0].name.replace('mask', 'mask_value'))
            mask_update = sess.graph.get_operation_by_name(gdp_mask_taylor[i][0].name.replace('mask', 'mask_update')[:-2])
            for j in range(gdp_mask_taylor[i][0].shape[0]):
                sess.run(mask_update, feed_dict={mask_index:j, mask_value:1})

        for i in range(FLAGS.taylor_step):
            tvs = sess.run([mt[1] for mt in gdp_mask_taylor])
            for g in range(len(gdp_mask_taylor)):
                mean_tvs[g] += tvs[g]
        
        for g in range(len(gdp_mask_taylor)):
            mean_tvs[g] /= FLAGS.taylor_step

        for t, mt in enumerate(gdp_mask_taylor):
            for i in range(tvs[t].shape[0]):
                ft = dict()
                ft['mask'] = mt[0]
                ft['index'] = i
                ft['tv'] = tvs[t][i]
                network_filters.append(ft)

        network_filters_sorted = sorted(network_filters, key=lambda a: -a['tv'])

        for i, ft in enumerate(network_filters_sorted):
            mask_index = sess.graph.get_tensor_by_name(ft['mask'].name.replace('mask', 'mask_index'))
            mask_value = sess.graph.get_tensor_by_name(ft['mask'].name.replace('mask', 'mask_value'))
            mask_update = sess.graph.get_operation_by_name(ft['mask'].name.replace('mask', 'mask_update')[:-2])

            if i < int(FLAGS.beta*len(network_filters)):
                sess.run(mask_update, feed_dict={mask_index:ft['index'], mask_value:1})
            else:
                sess.run(mask_update, feed_dict={mask_index:ft['index'], mask_value:0})

        layer_compression_ratio = tf.get_collection('LAYER_COMPRESSION_RATIO')
        lrc = sess.run(layer_compression_ratio)
        for i in range(len(layer_compression_ratio)):
            print(layer_compression_ratio[i].name[:-2], lrc[i])

        K.set_learning_phase(True)

    total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
    time_elapsed = time.time() - start_time

    if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(train_step_kwargs['logdir'],
                                      'tf_trace-%d.json' % np_global_step)
        logging.info('Writing trace to %s', trace_filename)
        file_io.write_string_to_file(trace_filename, trace)
        if 'summary_writer' in train_step_kwargs:
          train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                               'run_metadata-%d' %
                                                               np_global_step)

    if 'should_log' in train_step_kwargs:
        if sess.run(train_step_kwargs['should_log']):
            logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                   np_global_step, total_loss, time_elapsed)

    if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
    else:
        should_stop = False

    return total_loss, should_stop



def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as g:
        K.set_learning_phase(True)
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
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20 * FLAGS.batch_size,
                common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            label -= FLAGS.labels_offset

            train_image_size = FLAGS.train_image_size or network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - FLAGS.labels_offset)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels = batch_queue.dequeue()
            logits = network_fn(images)

            #############################
            # Specify the loss function #
            #############################
            slim.losses.softmax_cross_entropy(
                logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
            return None

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

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

        if FLAGS.save_max_keep != 5:
            saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=FLAGS.save_max_keep)
        else:
            saver = None

        # Set train step kwargs
        with tf.name_scope('train_step'):
            train_step_kwargs = {}
            if FLAGS.max_number_of_steps:
                should_stop_op = tf.greater_equal(global_step, FLAGS.max_number_of_steps)
            else:
                should_stop_op = tf.constant(False)
            train_step_kwargs['should_stop'] = should_stop_op
            if FLAGS.log_every_n_steps > 0:
                train_step_kwargs['should_log'] = tf.equal(
                    tf.mod(global_step, FLAGS.log_every_n_steps), 0)

            # GDP Taylor Value    
            gdp_mask_taylor = []
            var_ops = tf.get_collection('GDP_VAR')
            for var in var_ops:
                weight = var[0]
                bias = None
                if len(var) > 2:
                    bias = var[1]
                
                # get corresponding gradient ops
                weight_gradient = None
                for go in clones_gradients:
                    if go[1].name == weight.name:
                        weight_gradient = go[0]
                        break

                if bias is not None:
                    bias_gradient = None
                    for go in clones_gradients:
                        if go[1].name == bias.name:
                            bias_gradient = go[0]
                            break

                # calculate the corresponding taylor value
                filter_num = int(weight.shape[-1])
                if bias is not None:
                    taylor_value = tf.abs(tf.reduce_sum(weight * weight_gradient, axis=[0, 1, 2]) + bias * bias_gradient)
                else:
                    taylor_value = tf.abs(tf.reduce_sum(weight * weight_gradient, axis=[0, 1, 2]))

                # set mask update op
                mask = var[-1]

                gdp_mask_taylor.append((mask, taylor_value))
            
            train_step_kwargs['gdp_mask_taylor'] = gdp_mask_taylor

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            train_step_fn=train_step,
            train_step_kwargs=train_step_kwargs,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            init_fn=_get_init_fn(),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            saver=saver,
            sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
    tf.app.run()
