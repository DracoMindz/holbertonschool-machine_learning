#!/usr/bin/env python3
"""Defines a neural network"""


import numpy as np


class NeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, nodes):
        """nx is input values; nodes are num nodes in hidden layer"""
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not (int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """return weight vector for hidden layer"""
        return self.__W1

    @property
    def W2(self):
        """return weight vector output neuron"""
        return self.__W2

    @property
    def b1(self):
        """return bias for hidden layer"""
        return self.__b1

    @property
    def b2(self):
        """return bias for output neuron"""
        return self.__b2

    @property
    def A1(self):
        """return activated output hidden layer"""
        return self.__A1

    @property
    def A2(self):
        """return activated output for output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """calculates forward propagation of NN"""
        M = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1.0/(1.0 + np.exp(-M))
        self.__A2 = (np.dot(self.__W2, self.__A1) + self.__b2)
        self.__A2 = 1.0/(1.0 + np.exp(-1 * self.A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """cost of model using logistic regression"""
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """evaluate NN predictions"""
        return (self.forward_prop(X)[1].round().astype(int),
                self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass gradient descent on the NN"""
        dpW2 = np.dot(self.__W2.T, (A2 - Y)) * A1 * (1 - A1)
        self.__W1 = (self.__W1 - alpha * np.dot((A2 - Y), X.T) / X.shape[1])
        self.__W2 = (self.__W2 - alpha * np.dot(dpW2, A1.T) / A1.shape[1])
        self.__b1 = self.__b1 - alpha * (A2 - Y).mean(1, keepdims=True)
        self.__b2 = self.__b2 - alpha * dpW2.mean(1, keepdims=True)
