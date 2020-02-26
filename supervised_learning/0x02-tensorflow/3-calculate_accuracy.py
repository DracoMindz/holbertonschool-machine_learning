#!/usr/bin/env python3
"""function calculates accuracy of prediction"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    """

    prediction = tf.argmax(y_pred, 1)
    results = tf.argmax(y, 1)
    equal = tf.equal(prediction, results)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
