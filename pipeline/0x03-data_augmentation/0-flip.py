#!/usr/bin/env python3
"""
Function that flips an image horizontally
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def flip_image(image):
    """
    flips image horizontally
    :param image: 3D tf.tensor contains image to flip
    :return: flipped image
    """
    flipHoriz = tf.image.flip_left_right(image)
    return flipHoriz
