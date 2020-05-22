#!/usr/bin/env python3
"""
Function performs K-means on a dataset
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset
    :param X: np.ndarray, shape(n, d)
              containing dataset
    :param k: positive integer containing num of clusters
    :param iterations: 1000
    :return: C, clss or None, None on failure
    """

    centroids = initialize(X, k)
    if iterations < 1 or not isinstance(iteration, int):
        return None
    if centroid is None:
        return None, None
    klass = np.zaeros(X.shape[0], dtype=int)


def initialize(X, k):
    """
    Initialize K clusters, mult variable dist
    """
