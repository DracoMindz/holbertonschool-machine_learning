#!/usr/bin/env python3
"""calculates the softmax cross-entropy loss of a prediction"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    y is placeholder for labels of input data
    y_pred tensor containing network's rediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
