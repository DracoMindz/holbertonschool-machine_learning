#!/usr/bin/env python3
"""
function performs forward propagation over
convolutional layer of NN
"""

import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    forward propagation with convolution layer
    """

    """
    A_prev: numpy.ndarray with shape(m, h_prev, w_prev, c_prev)
            contains output of previous layer
        m: num of examples
        h_prev: height of prev layer
        w_prev: width of prev layer
        c_prev: num channels in prev layer
    W: numpy.ndarray shape(kh, kw, c_prev, c_new)
       contains kernels for convolution
        kh: filter height
        kw: filter weight
        c_prev: num of channels in prev layer
        c_new: num channels in output
    b: numpy.darray shape(1, 1, 1, c_new)
       contains biases applied to convolution
    activation: activation function applied to convolution
    padding: string either 'same' or 'valid' type padding
    stride: tuple shape(sh, sw)
       sh: stride for height
       sw: stride for width
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3] = W.shape[2] = b.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]
    ph = padding[0]
    pw = padding[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph == 0
        pw == 0
    if padding == 'same':
        ph =
        pw =
    