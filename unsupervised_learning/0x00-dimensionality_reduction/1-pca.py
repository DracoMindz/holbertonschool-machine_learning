#!/usr/bin/env python3
"""
function that peforms  PCA  on a dataset
"""

import numpy as np


def pca(X, ndim):
    """
    function performs PCA on a dataset
    :param X: numpy.ndarray, shape(n, d)
    :param n: num data points
    :param d: num dimensions in each point
    :param ndim: new dimensionality of the transformed X
    :param T: numpy.ndarray, shape(n, ndim) ndim is new dimensionality of
                transformed X
    :return: T
    """
    (m, n) = X.shape
    X = X - np.mean(X, axis=0)
    _, _, V = np.linalg.svd(X)
    W = V.T[:, ndim]
    return np.matmul(X, W)
