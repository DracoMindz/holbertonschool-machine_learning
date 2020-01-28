#!/usr/bin/env python3
"""Function to find size of Matrix"""


def matrix_shape(matrix):
    """Finds dimensions of matrix"""
    if len(matrix) == 0:
        return [0]
    m = matrix[:]
    size = [len(m)]
    while type(matrix[0]) is list:
        size.append(len(matrix[0]))
        matrix = matrix[0]
    return size
