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
    if iterations < 0 or not isinstance(iterations, int):
        return None
    if centroids is None:
        return None, None
    pt_assign = np.zeros(X.shape[0], dtype=int)
    while iterations > 0:
        previous = pt_assign.copy()
        update_assignedpts(centroids, X, pt_assign)

        #  iterate through the length of the centroids
        for c_indx in range(len(centroids)):
            assigned = np.where(pt_assign == c_indx, True, False)

        #  multivariate uniform distribution
            if assigned.sum() == 0:
                min_X = X.min(axis=0)
                max_X = X.max(axis=0)
                centroids[c_indx] = np.random.uniform(min_X, max_X,
                                                      (1, X.shape[1]))
                continue
            centroids[c_indx] = X[assigned, :].sum(axis=0) / assigned.sum()
        if (pt_assign == previous).all():
            break
        iterations -= 1
    update_assignedpts(centroids, X, pt_assign)
    return centroids, pt_assign


def update_assignedpts(centroids, X, pt_assign):
    """
    update points assignment to centroid
    """
    for c_indx, centroid in enumerate(centroids):
        for p_indx, point in enumerate(X):
            assigned_point = pt_assign[p_indx]
            # check if point assigned match centroid index
            if assigned_point == c_indx:
                continue
            aptcent_diff = (point - centroids[assigned_point])
            ptcent_diff = (point - centroid)
            assigned = pow(aptcent_diff, 2).sum()
            verify_pts = pow(ptcent_diff, 2).sum()
            if verify_pts < assigned:
                pt_assign[p_indx] = c_indx


def initialize(X, k):
    """
    Initialize K clusters, mult variable dist
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None
    d = X.shape[1]
    min_X = X.min(axis=0)
    max_X = X.max(axis=0)
    return np.random.uniform(min_X, max_X, size=(k, d))
