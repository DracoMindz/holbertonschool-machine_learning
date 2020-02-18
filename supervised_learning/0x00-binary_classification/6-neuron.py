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
        self.__W = np.random.normal(size=nx).reshape(nx, 1).T
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

    def evaluate(self, X, Y):
        """evaluates neuron's redictions"""
        self.forward_prop(X)
        return np.round(self.__A).astype(int), self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent on the nueron"""
        self.__W = (self.__W - alpha * np.dot(X, (A - Y).T).T / X.shape[1])
        self.__b = self.__b - alpha * (A - Y).mean()

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """trains the neuron"""
        if type(iterations) is not (int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not (float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
            i = i - 1
        return self.evaluate(X, Y)
