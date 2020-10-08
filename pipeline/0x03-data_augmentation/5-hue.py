#!/usr/bin/env python3
"""
Function changes the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    changes hue of image
    :param image: 3D tf.Tensor
    :param delta: amount of hue change
    :return: altered image
    """

    imHueChange = tf.image.adjust_hue(image, delta=delta, name=None)
    return imHueChange
