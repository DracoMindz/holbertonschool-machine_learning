#!/usr/bin/env python3
"""function that performs elelment-wise operations"""


def np_elementwise(mat1, mat2):
    """perform element-wise operations"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
