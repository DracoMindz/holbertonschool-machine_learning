#!/usr/bin/env python3
"""
Function Randomly shear image
"""

import tensorflow as tf


def shear_image(image, intensity):
    """
    randomly shears image
    :param image: 3D tf.tensot
    :param intensity: shear intensity
    :return: sheared image
    """
    imSheared = tf.keras.preprocessing.image.random_shear(
        image, intensity, row_axis=1, col_axis=2, channel_axis=0,
        fill_mode='nearest', cval=0.0, interpolation_order=1)
    return imSheared
