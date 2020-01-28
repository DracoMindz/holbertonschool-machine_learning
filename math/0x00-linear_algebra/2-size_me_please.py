#!/usr/bin/env python3
"""Function to find size of Matrix"""


def matrix_shape(matrix):
    """Finds dimensions of matrix"""
    if len(matrix) == 0:
        return [0]
    size = [len(matrix)]
    if type(matrix) is int:
        return
    else:
        size.append(len(matrix[0]))
        matrix = matrix[0]
        return size
