#!/usr/bin/env python3
"""function concatenates two matrices along an axis"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenate two matrices along axis"""
    return np.concatenate((mat1, mat2), axis)
