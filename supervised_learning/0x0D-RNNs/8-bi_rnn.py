#!/usr/bin/env python3
"""
Forward propagation for a BiDirectional RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    function performs forward propagation for a
    bidirectional RNN
    :param bi_cell: an instance of BidirectionalCell
            used for the forward propagation
    :param X: data to be used; np.ndarray; shape(t,m,i)
    :param h_0: initial hidden state in forward direction
                np.ndarray; shape(m,h)
    :param h_t: initial hidden state for backward direction
                np.ndarray; shape(m,h)
    Note: t: max num of time steps
          m: batch size
          i: dim of data
          h: dim of hidden state
    :return: H, Y
    Note: H: np.ndarray;
             contains: all concatenated hidden states
          Y: np.ndarray;
             contains: all of the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape
    H_forward = np.zeros((t+1, m, h))
    H_backward = np.zeros((t+1, m, h))

    # forward and backward hidden
    H_forward[0] = h_0
    H_backward[-1] = h_t

    # rang going forward and backward through RNN
    for step in range(t):
        H_forward[step+1] = bi_cell.forward(H_forward[step], X[step])
    # for b_step in range(t-1, -1, -1):
        H_backward[t-1-step] = bi_cell.backward(H_backward[t-step],
                                                X[t-1-step])

    # all concatenated hidden states
    H = np.concatenate((H_forward[1:], H_backward[0:-1]), axis=-1)
    # output with the concatenated hidden states
    Y = bi_cell.output(H)

    return H, Y
