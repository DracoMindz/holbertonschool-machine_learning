#!/usr/bin/env python3
"""function converts numeric label vector to one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is numpy.ndarray shape (m,) contains muneric labels
    m is num of examples
    classes is max num classes found in Y
    """
    b = np.zeros((Y.shape[0], classes + 1))
    if Y.any():
        for cl, m in enumerate(Y):
            b[cl][m] = 1
            return b
        else:
            return None
