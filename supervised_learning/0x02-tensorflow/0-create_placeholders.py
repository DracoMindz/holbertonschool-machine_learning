#!/usr/bin/env python3
"""function returns two placholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    nx is num feature columns in data
    classes is num classes in classifier
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return (x, y)
