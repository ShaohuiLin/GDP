# coding: utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

class MaskConv2D(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name="conv2d",
                 is_training=True,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.is_training = is_training
        self.layername = name
        super(MaskConv2D, self).__init__(name=self.layername, **kwargs)

    def build(self, input_shape):
        # with tf.variable_scope(self.layername):
        self.kernel = self.add_weight(name='kernel', shape=[self.kernel_size[0], self.kernel_size[1],
                                                            int(input_shape[3]), self.filters],
                                        initializer=initializers.get(self.kernel_initializer), 
                                        regularizer=self.kernel_regularizer,
                                        trainable=self.is_training)
        
        self.mask = self.add_weight(name='mask', shape=[self.filters],
                                    initializer=tf.initializers.ones(), trainable=False)
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=[self.filters],
                                        initializer=initializers.get(self.bias_initializer),
                                        trainable=self.is_training)

            tf.add_to_collection('GDP_VAR', [self.kernel, self.bias, self.mask])
        else:
            tf.add_to_collection('GDP_VAR', [self.kernel], self.mask)

        index = tf.placeholder(dtype=tf.int32, shape=None, name='mask_index')
        value = tf.placeholder(dtype=tf.float32, shape=None, name='mask_value')
        update_op = tf.scatter_update(self.mask, index, value, name='mask_update')

        tf.add_to_collection('LAYER_COMPRESSION_RATIO', tf.reduce_mean(self.mask, name="compression_ratio"))

        super(MaskConv2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        out = tf.nn.conv2d(inputs, self.kernel * self.mask, strides=[1, self.strides[0], self.strides[1], 1],
                           padding=self.padding)
        if self.use_bias:
            out = out + self.bias * self.mask
        if self.activation is not None:
            out = activations.get(self.activation)(out)

        return out

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(activations.get(self.activation)),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(initializers.get(self.kernel_initializer)),
            'bias_initializer': initializers.serialize(initializers.get(self.bias_initializer)),
            'kernel_regularizer': regularizers.serialize(regularizers.get(self.kernel_regularizer)),
            'bias_regularizer': regularizers.serialize(regularizers.get(self.bias_regularizer)),
            'activity_regularizer': regularizers.serialize(regularizers.get(self.activity_regularizer)),
            'kernel_constraint': constraints.serialize(constraints.get(self.kernel_constraint)),
            'bias_constraint': constraints.serialize(constraints.get(self.bias_constraint))
        }
        base_config = {"name": self.layername}
        return dict(list(base_config.items()) + list(config.items()))

