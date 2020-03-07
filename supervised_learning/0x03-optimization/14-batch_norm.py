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
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    m = tf.layers.Dense(units=n, activation=None,
                          kernel_initializer=init)
    beta = (tf.nn.get_variable("beta", [n],
                               initializer=tf.zeros_initializer(),
                               trainable=True))
    gamma = (tf.nn.get_variable("gamma", [n],
                                initializer=tf.ones_initializer(),
                                trainable=True))
    batch_mean, batch_var = tf.nn.moments(m(prev), axes=[0])
    norm_layer = (tf.nn.batch_normalization(m(prev), mean=batch_mean,
                                            variance=batch_var, offset=beta,
                                            scale=gamma,
                                            variance_epsilon=1e-8))
    return activation(norm_layer)
