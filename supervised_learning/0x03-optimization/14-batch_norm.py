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
    init = (tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    mm = (tf.layers.Dense(units=n, activation=None,
                          kernel_initializer=init))
    gamma = (tf.nn.get_variable("gamma", [n],
                                initializer=tf.ones_initializer(),
                                trainable=True))
    beta = (tf.nn.get_variable("beta", [n],
                               initializer=tf.zeros_initializer(),
                               trainable=True))
    batch_mean, batch_var = (tf.nn.moments(x(prev), axes=0))
    norm_layer = (tf.nn.batch_normalization(x(prev), mean=batch_mean,
                                            variance=batch_var, offset=beta,
                                            scale=gamma,
                                            variance_epsilon=1e-8))
    act_norm_layer = activation(norm_layer)
    return act_norm_layer
