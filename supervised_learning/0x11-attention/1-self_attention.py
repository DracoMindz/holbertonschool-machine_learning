#!/usr/bin/env python3
"""
Class SelfAttention
"""
import numpy
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention Class
    """
    def __init__(self, units):
        """
        Initiailize variables
        :param units: int representing num of hidden units
        Public Instances
        :W - Dense layer with units=units,
             applied to the previous decoder hidden state
        :U - Dense layer with units=units
             applied ro the encoder hidden states
        :V - Dense layer units=1
            applied to the tanh of the sum of outputs of W & U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Public instance method
        :param s_prev: tensor shape(batch, units)
                        contains previous decoder hidden
        :param hidden_states: tensor of shape(batch, input_seq_len, units)
                            contains output of encoder
        :return: context, weights
            context: tensore shape(batch, units)
            contains context vector for the decoder
            weights: tensor shape(batch, input_seq_len, 1)
            contains attention weights
        """
        decW = self.W(s_prev)
        decW = tf.expand_dims(decW, axis=1)
        encU = self.U(hidden_states)
        outV = self.V((tf.nn.tanh(decW +encU)))
        weights = tf.nn.softmax(outV, axis=1)
        context = tf.reduce_sum((weights * hidden_states), axis=1)
        return context, weights

