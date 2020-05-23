#!/usr/bin/env python3
"""
calculates the probability density function of a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    caalculates the probability density function
    :param X: np.ndaray, (n, d) data points
    :param m: np.ndarray, (d,) mean of distribution
    :param S: np.ndarray, (d,d) covariance of distribution
    :return: P or None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or X.shape[1] != S.shape[1]:
        return None
    # n, d = X.shape
    g_pdf = np.empty(X.shape[0])
    for index, xndx in enumerate(X):
        xm = (xndx - m)
        d = X.shape[1]
        g_pdf[index] = (np.exp(np.dot(np.dot((xm),
                                             np.linalg.inv(S)),
                                      xm.T) / -2) /
                        np.sqrt(pow(2 * np.pi, d) * np.linalg.det(S)))
    P = np.where(g_pdf > 1e-300, g_pdf, 1e-300)
    return P
