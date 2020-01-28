#!/usr/bin/env python3
"""Function to find size of Matrix"""


def matrix_shape(matrix):
    """Finds dimensions of matrix"""

    size = []

    if (len(matrix) > 0):
        size.append(len(matrix))
        return (size)
    else:
        return [0]
