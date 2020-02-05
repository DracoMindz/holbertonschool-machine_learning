#!/usr/bin/env python3
"""function slices a matrix along a soecific axes"""


import numpy as np


def np_slice(matrix, axes={}):
    """slice matrix along axis"""
    for axis, key in axes.items():
        chop = [slice(None)] * (axis + 1)
    return matrix[tuple(chop)]
