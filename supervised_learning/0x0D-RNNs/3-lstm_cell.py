#!/usr/bin/env python3
"""
LSTMCell class
"""

import numpy as np


class LSTMCell:
    def __init__(self, i, h, o):
        """
        class constructor
        :param i: dim of data
        :param h: dim of hidden state
        :param o: dim of outputs
        Note: Creates public attributes:
            Wf and bf: for forget gate
            Wu and bu: for update gate
            Wc and bc: for intermediate cell state
            Wo and bo: for output gate
            Wy and by: for outputs
        Note: Weights: initialized using random normal distribution
                        used on the right side for matrix mult
        Note: Biases: intitialized as zeros
        """
        # public attribute weights; initialized
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))

        # public attribute bias; initialized
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        functino performs forward propagation for one time step
        :param h_prev: np.ndarray, shape(m, h);
                contains: previos hidden state
        :param c_prev: np.ndarray, shape(m, h);
                contains: prev cell state
        :param x_t: np.ndarray, shape(m, i);
                contains: data input for cell
        Note: h_next: next hidden state
              c_next: next cell state
              y: output of the cell
        :return: H_next, c_next, y
        """
        hidConInput = np.concatenate((h_prev, x_t), axis=1)

        # definition of forget gate using Wf and bf
        forgetGate = self.sigmoid(np.matmul(hidConInput, self.Wf) + self.bf)

        # definition of update gate using Wz and bz
        upGate = self.sigmoid(np.matmul(hidConInput, self.Wu) + self.bu)

        # definition of output gate using Wr and br
        outGate = self.sigmoid(np.matmul(hidConInput, self.Wo) + self.bo)

        # definition of intermediate cell state using Wc, bc, tanh
        i_cellState = np.tanh(np.matmul(hidConInput, self.Wc) + self.bc)

        # definition of cell state
        c_next = forgetGate * c_prev + upGate * i_cellState

        # definition of next hidden state
        h_next = outGate * np.tanh(c_next)

        # compute prediction of LTSM Cell
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
