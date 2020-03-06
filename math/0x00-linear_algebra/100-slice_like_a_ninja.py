#!/usr/bin/env python3
"""function slices a matrix along a soecific axes"""


import numpy as np


def np_slice(matrix, axes={}):
    """slice matrix along axis"""
    chop = [slice(None)] * (max(axes) + 1)
    for axis, key in axes.items():
        chop[axis] = slice(*key)
    return matrix[tuple(chop)]
