#!/usr/bin/env python3
"""function converts one-hot matrix to vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot is encoded numpy.ndarray w/ shape
    (classes, m)
    """

    if len(one_hot.shape) != 2 or len(one_hot) == 0:
        return None
    if type(one_hot) is not np.ndarray:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
