#!/usr/bin/env python3
"""
Fuction performs forward propagation for simple RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation
    :param rnn_cell: instance of RNNCell, used for forward propagation
    :param X: data to be used, np.ndarray, shape(t, m, i)
    :param h_0: initial hidden state, npndarray, shape(m, h)
    Note: t = max num of time steps
          m = batch size
          i = dimensionality of the data
    Note: h = dimensionality of hidden state
    :return: H, Y
        Note:   H = np.ndarray, contains hidden states
                Y = np.ndarray, contains all the outputs
    """

    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t+1, m, h))
    Y = []
    H[0] = h_0
    for step in range(t):
        H[step+1], y_layer = rnn_cell.forward(H[step], X[step])
        Y.append(y_layer)
    # reshape or asarray to return np.ndarray
    Y = np.asarray(Y)
    return H, Y
