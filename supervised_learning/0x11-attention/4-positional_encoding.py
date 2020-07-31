#!/usr/bin/env python3
"""
function calculates the positional encoding got a transformer
"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    calculates positional encoding for a transformer
    :param max_seq_len: int representing the max sequence length
    :param dm: model depth
    :return: numpy.ndarray shape(max_seq_len, dm)
             containing positional encoding vectors
    """
    posEncoding = np.zeros([max_seq_len, dm])

    # Transformer Model fo Language Understanding
    # getting the angles
    for i in range(dm):
        for pos in range(max_seq_len):
            posEncoding[pos, i] = pos / np.power(10000, (2 * (i // 2)) / (dm))
            # angleRate = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
            # posEncoding = pos * angleRate
    # use sin for even indices in the array; 2i
    posEncoding[:, 0::2] = np.sin(posEncoding[:, 0::2])
    # use cos for odd indices in the array; 2i + 1
    posEncoding[:, 1::2] = np.cos(posEncoding[:, 1::2])

    return posEncoding
