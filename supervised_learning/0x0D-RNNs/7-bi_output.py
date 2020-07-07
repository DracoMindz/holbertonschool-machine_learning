#!/usr/bin/env python3
"""
Update BidirectionalCell w/ backward
"""

import numpy as np


class BidirectionalCell:
    def __init__(self, i, h, o):
        """
        class constructor
        :param i: dim of data
        :param h: dim of hidden states
        :param o: dim outputs
        Notes: Public Instances:
                Whf and bhf: for the hidden states
                            forward direction
                Whb and bhb: for hiden states
                            backward direction
                Wy and by: for outputs
        Notes: Weights: initialized using random normal dist
                        in order listed
                        : used on the right side for matrix mult
        Notes: Biases: initialize as zeros
        """
        # public attributes weights; initialized
        self.Whf = np.random.normal(size=(i+h, h))
        self.Whb = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(2*h, o))

        # public attributes baises; intialized
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        softmax function
        :return: values for scores in x
        """
        xMax = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - xMax)
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        public instance method: calculates hidden state in
        forward direction for one time stop
        :param h_prev: np.ndarray; shape(m, h);
                      contains: prev hidden state
        :param x_t: np.ndarray; shape(m, i);
                      contains: data imput for the cell
        Note: m: batch size for the data
        :return: h_next
        Note: h_next: next hidden state
        """
        h_matrix = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((np.matmul(h_matrix.T, self.Whf)) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        public instance method calculates hidden state in backward direction
        for one time step
        :param h_next: np.ndarray; shape(m, h); contains: next hidden state
        :param x_t: np.ndarray; shape(m, i); contains: data input for the cell
        Notes: m: batch size for the data
        :return: h_prev
        Note: h_prev: previous hidden state
        """
        h_matrix = np.concatenate((h_next.T, x_t.T), axis=0)
        h_prev = np.tanh(np.matmul(h_matrix.T, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
        public instance method calculates all outputs for RNN
        :param H: np.ndarray; shape(t, m, 2*h)
        Note: t: num of time steps
              m: batch size for the data
              h: dim of hidden states
        :return: Y
        Notes: Y: outputs
        """
        # output use softmax and Wy and by
        Y = self.softmax((np.matmul(H, self.Wy)) + self.by)
        return Y
