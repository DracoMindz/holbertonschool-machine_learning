#!/usr/bin/env python3
"""
GRUCell class
"""

import numpy as np


class GRUCell:
    def __init__(self, i, h, o):
        """
        class constructor
        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of outputs
        Note: creates public instancves for :
            Wz and bz: for the update gate
            Wr and br: for the reset gate
            Wh and bh: for intermediate hidden state
            Wy and by: for the output
        Note: Weights initialized using random normal distribution
                in listed order
              Weights used on the right side for matrix nultiplication
        Note: biases initialized as zeros
        """
        self.Wz = np.random.normal(size=(h+i, h))
        self.Wr = np.random.normal(size=(h+i, h))
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        softmax function
        :return: values for scores in x
        """
        xMax = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - xMax)
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def sigmoid(self, x):
        """
        sigmoid function
        :return: sigmoid values for x
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """
        public instance method
        performs forward propagatiuon for one time step
        :param h_prev: np.ndarray, shape(m, h), contains previous hidden state
        :param x_t: np.ndarray, shape(m, i), contains data input for the cell
            Note: m is the batch size for the data
                  i (from GRUCell) dims of data
                  h (from GRECell) dims of hidden state
            Note: h_next: the next hidden state
            Note: y: output of cell
        :return: h_next, y
        """
        hidConInput = np.concatenate((h_prev, x_t), axis=1)
        # definition of update gate using Wz and bz
        upGate = self.sigmoid(np.matmul(hidConInput, self.Wz) + self.bz)

        # definition of reset gate using Wr and br
        reGate = self.sigmoid(np.matmul(hidConInput, self.Wr) + self.br)

        gate_hidConInput = np.concatenate((reGate * h_prev, x_t), axis=1)
        # definition of h_prop
        h_prop = np.tanh(np.matmul(gate_hidConInput, self.Wh) + self.bh)

        # definition of h_next, the next hidden state
        h_next = upGate * h_prop * (1 - upGate) * h_prev

        # definition output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
