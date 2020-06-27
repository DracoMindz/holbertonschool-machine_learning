#!/usr/bin/env pyhton3
"""
Function that creates input for the generator.
"""

import numpy as np
import tensorflow as tf


def sample_Z(m, n):
    """
    creates input for generator
    :param m: num of samples that should be generated
    :param n: num of dimensions of each sample
    Note: all samples should be taken from a random uniform distribution
            within range [-1, 1]
    :return: Z, a numpy.ndarray, shape(m, n) contaons: uniform samples
    """
    Z = np.random.uniform(-1, 1, size=[m, n])
    return Z
