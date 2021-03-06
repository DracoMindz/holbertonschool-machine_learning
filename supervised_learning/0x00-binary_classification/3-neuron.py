#!/usr/bin/env python3
"""Class defines single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive")
        self.__W = np.ndarray((1, nx))
        self.__W[0] = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """return weights"""
        return self.__W

    @property
    def b(self):
        """returns bias"""
        return self.__b

    @property
    def A(self):
        """return activation output"""
        return self.__A

    def forward_prop(self, X):
        """Updates the private attribute __A"""
        M = np.matmul(self.__W, X) + self.__b
        self.__A = 1.0/(1.0 + np.exp(-M))
        return self.__A

    def cost(self, Y, A):
        """Cost of Model using logistic regression"""
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
