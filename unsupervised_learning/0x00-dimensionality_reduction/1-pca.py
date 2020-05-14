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
    (n, d) = X.shape
    m = mean(X, axis=1)
    mid_X = X - m
    cov_X = np.dot(mid_X.T, mid_X) / (n-1)

    # get eigenvalues  and eigenvectors
    eigVal, eigVec = np.linalg.eig(cov_X)
    idx = np.argsort(eigVal)[::-1]
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]

    eigVal = eigVal[:ndim]
    inv_eigVal = (-1) * eigVec[:, :ndim]
    return float64(np.matmul(mid_X, inv_eigVal))
