#!/usr/bin/env python3
"""
Class Transformer
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ class"""
    def __init__(self, dm, h):
        """
        class constructor
        param:: dm: int representing dims of the model
        param:: h: int representing num of heads
        Note:: dm: divisible by h
        Public Instances
        h: num of heads
        dm: dims of the model
        depth: depth of each atten head
        Wq: Dense layer w/ dm units
            used to generate query matrix
        Wk: Dense layer w/ dm units
            used to generate key matrix
        Wv: Dense layer w/ dm units
            used to generate value matrix
        linear: Dense layer w/ dm units
            used to generate attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm // self.h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def splitHeads(self, m, batch):
        """
        split last dim shape(self.h, self.depth)
        transpose result shape(batch, -1, self.h, self.depth)
        """
        m = tf.reshape(m, (batch, -1, self.h, self.depth))
        return tf.transpose(m, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Public Instance Method
        param:: Q: tensor shape (batch, seq_len_q, dk)
                contains input to generate the query matrix
        param:: K: tensor shape (batch, seq_len_v, dk)
                contains input to generate the key matrix
        param:: V: tensor shape (batch, seq_len_v, dv)
                contains input to generate the value matrix
        Note: mask is always None
        Returns: output, weights
                output: tensor with last two dims (..., seq_len_q, dm)
                        contains scaled dot product attention
                weights: tensor with last three dims
                        (..., h, seq_len_q, seq_len_v)
                        contains attention weights
        """
        # batch size
        batch = tf.shape(K)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # split dims(self.h, batch)
        # transpose shape(batch, -1, self.h, self.depth)
        Q = self.splitHeads(Q, batch)
        K = self.splitHeads(K, batch)
        V = self.splitHeads(V, batch)

        # use task 5 here
        # output is scaled attention, weights is attention weights
        output, weights = sdp_attention(Q, K, V, mask)

        # transpose output, reshape output,
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)

        return output, weights
