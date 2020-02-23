#!/usr/bin/env python3
"""function converts numeric label vector to one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is numpy.ndarray shape (m,) contains muneric labels
    m is num of examples
    classes is max num classes found in Y
    """
    if len(Y) == 0:
        return None
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    else:
        b = np.zeros((classes, Y.shape[0]))
        for cl, m in enumerate(Y):
            b[m][cl] = 1
        return b
