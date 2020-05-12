#!/usr/bin/env python3
"""Multivariate Normal Distribution Class"""


import numpy as np


class MultiNormal:
    """Multivariate Normal Distribution Class"""
    def __init__(self, data):
        """
        class constuctor
        :param data: np.ndarray, shape(d, n)
        :param n: num data points
        :param d: num dimensions in each data point
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        """
        set the public instance variables
        """
        self.mean = ((np.mean(data, axis=1)).reshape(data.shape[0], 1))
        X = data - self.mean
        self.cov = ((np.dot(X, X.T)) / (data.shape[1] - 1))
