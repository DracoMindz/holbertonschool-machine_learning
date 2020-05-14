#!/usr/bin/env python3
"""
function performs PCA on a dataset
"""

import numpy as np


def pca(X, var=0.95):

    """
    function performs PCA on a dataset
    :param X: numpy.ndarray, shape(n, d)
    :param n: num data points
    :param d: num dimensions in each point
    :param var: fraction of variance the PCA transformation should maintain
    :param W: numpy.ndarray, shape(d, nd) nd is new dimensionality of
                transformed X
    :return: weights matrix; maintains var fraction of X orig variance
    """
    # compute covariance matrix
    (n, d) = X.shape
    sigma = 1.0 / d * np.dot(X, X.T)

    # get eigenvectors and eigenvalues
    U, s, V = np.linalg.svd(sigma, full_matrices=True)

    # compute r  features and retain varieance
    s_sum = np.sum(s)
    var_x = np.array([np.sum(s[: i + 1]) / s_sum * 100.0 for i in range(n)])
    r = len(var_x[var_x < (var * 100)])

    # projected reduced dimensional features
    W = []
    red_U = U[:, : r]
    W = (-1 * red_U)
    return W
