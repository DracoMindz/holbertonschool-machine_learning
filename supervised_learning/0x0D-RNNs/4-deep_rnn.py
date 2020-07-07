#!/usr/bin/env python3
"""
Deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    function performs forward propagation for a deep RNN
    :param rnn_cells: list of RNNCell instances of length l
            used for forward prpagation
    :param X: data to be used, np.ndarray, shape(t, m, i)
    :param h_0: innitial hidden state, np.ndarray, shape(l, m, h)
    Note: layers: num of layers, layers
          timeSteps: max num of time steps
          m: batch size
          i: dim of data
          h: dim of hiddedn state
          H: np.ndarray, contains: hidden states
          Y: np.ndarray,  contains: all of the outputs
    :return: H, Y
    """
    timeSteps, m, i = X.shape
    layers, m, h = h_0.shape
    H = np.zeros((timeSteps+1, layers, m, h))
    Y = []
    H[0] = h_0

    for step in range(timeSteps):
        for layr in range(layers):
            if layr == 0:
                hid, y_layer = rnn_cells[layr].forward(H[step, layr], X[step])
            else:
                hid, y_layer = rnn_cells[layr].forward(H[step, layr], hid)
            H[step+1, layr, :, :] = hid
        Y.append(y_layer)
    # reshape or asarray to return np.ndarray
    Y = np.asarray(Y)
    return H, Y
