#!/usr/bin/env python3
"""Class defines a single neuron performing binary classification"""


import numpy as np


class Neuron:
    """Class defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.ndarray((1, nx))
        self.W[0] = np.random.normal(size=nx)
        self.b = 0
        self.A = 0
