#!/usr/bin/env python3
"""function creates the training operation for the network"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss is the loss of the network’s prediction
    alpha is the learning rate
    """

    opti = tf.train.GradientDescentOptimizer(alpha)
    return opti.minimize(loss)
