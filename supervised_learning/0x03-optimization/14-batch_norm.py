#!/usr/bin/env python3
"""
function creates  a batch normalization layer for NN
in tensorflow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev: activated output of previous layer
    n: num of nodes in layer to be created
    activation: activation function used on output of layer
    """
    norm_layer = tf.nn.batch_normalization(prev, n)
    act_norm_layer = activation(norm_layer)
    return act_norm_layer
