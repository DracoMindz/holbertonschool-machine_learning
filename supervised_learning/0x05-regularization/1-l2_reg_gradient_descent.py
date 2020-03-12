#!/usr/bin/env python3
"""
function updates the weights and biases of a neural network 
using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y: one-hot numpy.ndarray shape (classes, m) contains the correct labels
    weights: dictionary of weights and biases of NN
    cache: dictionary of outputs each layer of NN
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: num layers of network
    """
