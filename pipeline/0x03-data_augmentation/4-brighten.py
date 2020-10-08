#!/usr/bin/env python3
"""
Function Randomly changes the brightness of an image
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    changes the brightness of an image
    :param image: 3D tf.tensor
    :param max_delta: max amount to brighten image
    :return: altered image
    """

    imBright = tf.image.random_brightness(image,
                                          max_delta=max_delta, seed=None)
    return imBright
