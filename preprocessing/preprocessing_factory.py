#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import lenet_preprocessing
from preprocessing import vgg_preprocessing

slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):
  """Returns preprocessing_fn(image, height, width, **kwargs).

  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    preprocessing_fn: A function that preprocessing a single image (pre-batch).
      It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).

  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
      'cifarnet': cifarnet_preprocessing,
      'inception': inception_preprocessing,
      'inception_v1': inception_preprocessing,
      'inception_v2': inception_preprocessing,
      'inception_v3': inception_preprocessing,
      'inception_v3_prune': inception_preprocessing,
      'inception_v3_eightbit': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'lenet': lenet_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'nasnet_mobile': inception_preprocessing,
      'nasnet_large': inception_preprocessing,
      'resnet_v1_50': vgg_preprocessing,
      'resnet_v1_50_prune': vgg_preprocessing,
      'resnet_v1_50_eightbit': vgg_preprocessing,
      'resnet_v1_101': vgg_preprocessing,
      'resnet_v1_152': vgg_preprocessing,
      'resnet_v1_200': vgg_preprocessing,
      'resnet_v2_50': vgg_preprocessing,
      'resnet_v2_101': vgg_preprocessing,
      'resnet_v2_152': vgg_preprocessing,
      'resnet_v2_200': vgg_preprocessing,
      'vgg': vgg_preprocessing,
      'vgg_a': vgg_preprocessing,
      'vgg_16': vgg_preprocessing,
      'vgg_16_bnn': vgg_preprocessing,
      'vgg_16_sparse': vgg_preprocessing,
      'vgg_16_prune': vgg_preprocessing,
      'vgg_16_pruned': vgg_preprocessing,
      'vgg_16_eightbit': vgg_preprocessing,
      'vgg_16_admm': vgg_preprocessing,
      'vgg_16_aulm': vgg_preprocessing,
      'vgg_19': vgg_preprocessing,
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, output_height, output_width, is_training=is_training, **kwargs)

  return preprocessing_fn
