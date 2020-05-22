#!/usr/bin/env python3
"""
Initialize cluster centroids for K-means
"""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroid for K-means
    :param X: np.ndarray, shape(n, d) contains data set to be used for K-means
    :param n: num of data points
    :param d: num dimensions for each data point
    :param k: pos integer containing the num clusters
    :return: np.ndarray, of shape (k, d)
    """

    n = X.shape[0]
    d = X.shape[1]

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k < 1:
        return None
    if k > n or d < 1:
        return None
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))