#!/usr/bin/env python3
"""
RNNCell Class
"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """
        class condtructor
        :param i: dim of the data
        :param h: dim of hidden state
        :param o: dim of outputs
        Note: create public attributes Wh, Wy, bh, by
        Note: Wh and bh: for the concatenated Hidden state
                and input
              Wy and by: for the output
        Note: Weights: initialized using fandom normal distribution
                        in listed order
        Note: Biases: intialized as zeros
        return: public instances
        """
        # instance variables
        # self.i = i
        # self.h = h
        # self.o = o

        # create public attributes
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
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

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev: np.numpy, shape (m, h), contains prev hideend state
        :param x_t: np.numpy, shape(m. i), contains data input for the cell
        Note: m: batch size for the data
        NOte: h_next next hidden state
        Note: y  is the output of hte cell
        :return: h_next, y
        """
        hidden_con = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((np.matmul(hidden_con.T, self.Wh)) + self.bh)
        y = self.softmax((np.matmul(h_next, self.Wy)) + self.by)
        return h_next, y
