#!/usr/bin/env python3
"""
Functioin that creates a simple generator network for MINST digits
"""

import numpy as np
import tensorflow as tf


def generator(Z):
    """
    fcreates a simple generatfor network
    :param Z: tf.tensor, contains: input to generator network
    Note: network should have two layers
        1st layer 128 nodes, relu activiation, name= layer_1
        2nd 784 nodes, sigmoid activation, name=,ayer_2
    Note: All variables in network: have scope generator with
            reuse=tf.AUTO_REUSE
    :return: X, a tf.tensor containing the generated image
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.Dense(units=128, name='layer_1',
                                  activation=tf.nn.relu)
        X = tf. layers.Dense(units=784, name='layer_2',
                             activation=tf.nn.sigmoid)(layer_1)

        return X
