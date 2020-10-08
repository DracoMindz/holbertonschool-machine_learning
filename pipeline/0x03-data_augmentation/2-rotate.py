#!/usr/bin/env python3
"""
function rotates an image by 90 degrees counter-clockwise
"""


import tensorflow as tf


def rotate_image(image):
    """
    rotates image by 90 degrees counter clockwise
    :param image: 3D tf.Tensor
    :return: rotated image
    """

    imRotated = tf.image.rot90(image, k=1)
    return imRotated
