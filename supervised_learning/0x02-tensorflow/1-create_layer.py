#!/usr/bin/env python3
"""function to create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev is tensor output of prev layer
    n is num of nodes in layer to create
    activation is the function the layers should use
    """

    initialz = (tf.contrib.layers.
                variance_scaling_initializer(mode="FAN_AVG"))
    return tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=initialz)(prev)
