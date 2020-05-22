#!/usr/bin/env python3
"""
Initialize cluster centroids for K-means
"""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroid for K-means
    :param X: np.ndarray, shape(n, d)
                contains data set used for K-means
    :param n: num of data points
    :param d: num dimensions for each data point
    :param k: pos integer containing the num clusters
    :return: np.ndarray, of shape (k, d)
    """
    d = X.shape[1]
    n = X.shape[0]
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2):
        return None
    if (not isinstance(k, int) or k < 1 or k > n):
        return None
    min_X = np.amin(X, axis=0)
    max_X = np.amax(X, axis=0)
    return np.random.uniform(min_X, max_X, (k, d))
