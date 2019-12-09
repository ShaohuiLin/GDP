# coding:utf-8

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import regularizers
from utils.nnUtils_for_gdp import MaskConv2D

def net(inputs, num_classes, weight_decay, is_training=False):
    # Block 1
    with tf.variable_scope("conv1"):
        x = MaskConv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = MaskConv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    with tf.variable_scope("conv2"):
        x = MaskConv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    with tf.variable_scope("conv3"):
        x = MaskConv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    with tf.variable_scope("conv4"):
        x = MaskConv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    with tf.variable_scope("conv5"):
        x = MaskConv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaskConv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = Conv2D(4096, (7, 7), activation="relu", padding="valid", name='fc6', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(rate=0.3)(x, training=is_training)
    x = Conv2D(4096, (1, 1), activation="relu", padding="valid", name='fc7', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(rate=0.3)(x, training=is_training)
    x = Conv2D(num_classes, (1, 1), padding="valid", name='fc8', kernel_regularizer=regularizers.l2(weight_decay))(x)
    y = Flatten(name='fc8/squeezed')(x)

    return y