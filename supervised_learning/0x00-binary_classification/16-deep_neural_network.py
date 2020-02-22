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
        if type(layers) is not (list):
            raise TypeError("layers must be a list of positive integers")
        for i_layer in layers:
            """ i_layer reps elements in layers"""
            if type(i_layer) is not (int):
                raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {"W1": np.random.randn(layers[0], nx) *
                        np.sqrt(2 / nx),
                        "b1": np.zeros((layers[0], 1))}
        for i_layer, size in enumerate(layers[1:], 2):
            mWts = "W" + str(i_layer)
            self.weights[mWts] = (np.random.randn(size, layers[i_layer - 2]) *
                                  np.sqrt(2 / layers[i_layer - 2]))
            mB = "b" + str(i_layer)
            self.weights[mB] = np.zeros((layers[i_layer - 1], 1))
