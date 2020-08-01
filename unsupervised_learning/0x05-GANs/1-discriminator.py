#!/usr/bin/env python3
"""
Function that creates discriminator network
"""

import numpy as np
import tensorflow as tf


def discriminator(X):
    """
    create discriminator network
    :param X: tf.tensor contains: input to yjr discriminator network
    Note: Network has two layers
        1st layer: 128 nodes, relu activation, name= layer_1
        2nd layer: 1 node, sigmoid activation, name= layer_2
    All variables in network have scope discriminator with reuse=tf.AUTO_REUSE
    :return: Y, a tf.tensor containing classification made by discriminator
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.dense(units=128, name='layer_1',
                                  activation=tf.nn.relu)(X)
        Y = tf.layers.dense( units=1, name='layer_2',
                            activation=tf.nn.sigmoid)(layer_1)

        return Y
