#!/usr/bin/env pythons
"""
function creates training op for NN in tf
using RMSProp optimization algorithm
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    loss: loss of the network
    alpha: learning rate
    beta2: RMSProp weight
    epsilon: small number to avoid division by zero
    """

    return (tf.train.RMSPropOptimizer.
            __init__(learning_rate=alpha, decay=beta2, epsilon=epsilon).
            minimize(loss))
