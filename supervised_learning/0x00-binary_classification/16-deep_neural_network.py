#!/usr/bin/env python3
"""deep NN performing binary classififcation"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network Class"""

    def __init__(self, nx, layers):
        """nx is number of input values"""
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        """layers list reping num nodes in each layer"""
        if type(layers) is not (list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.nx = nx
        self.cache = {}
        for i_lyr in range(self.L):
            if type(layers[i_lyr]) is not (int) or layers[i_lyr] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights = {"W1": np.random.randn(layers[0], nx) *
                            np.sqrt(2 / nx),
                            "b1": np.zeros((layers[i_lyr], 1))}
            if type(layers[i_lyr]) is (int) and layers[i_lyr] > 1:
                mWts = "W" + str(i_lyr + 1)
                mB = "b" + str(i_lyr + 1)
                self.weights[mWts] = (np.random.randn(layers[i_lyr - 1], nx)
                                      * np.sqrt(2 / nx))
                self.weights[mB] = np.zeros((layers[i_lyr - 1], 1))
