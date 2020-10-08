#!/usr/bin/env python3
"""
crop image
"""


import tensorflow as tf


def crop_image(image, size):
    """
    crops image
    :param image: 3D tf.tensor
    :param size: tuple containing size
    :return: image
    """

    imCrop = tf.image.random_crop(image, size=size, seed=None, name=None)
    return imCrop
