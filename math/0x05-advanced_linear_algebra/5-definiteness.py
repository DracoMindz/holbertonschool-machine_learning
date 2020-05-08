#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    calculates definiteness of a matrix
    :param matrix: list of lists
    :return: the string: Positive definite,
    Positive semi-definite, Negative semi-definite,
    Negative definite, or Indefinite
    """

    # check if np.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    # check if symmetrical
    if not np.all(np.transpose(matrix) == matrix):
        return None
    if len(matrix.shape) != 2:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    definite = (np.linalg.eigvals(matrix))

    if all(definite == 0):
        return None
    if all(definite > 0):
        return "Positive definite"
    if all(definite < 0):
        return "Negative definite"
    if any(definite > 0) and any(definite == 0):
        return "Positive semi-definite"
    if any(definite < 0) and any(definite == 0):
        return "Negative semi-definite"
    elif not (any(definite < 0)
              and any(definite == 0) and any(definite > 0)):
        return "Indefinite"
    else:
        return "None"


def transpose_matrix(matrix):
    """
    transpose given matrix
    :param matrix: list of lists
    :return: transposed matrix
    """
    return [[row[m] for row in matrix] for m in range(len(matrix[0]))]
