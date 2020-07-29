#!/usr/bin/env python3
"""
Create RNNDecoder Class
"""

import numpy
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder Class
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        :param vocab: int representing size of output vocabulary
        :param embedding: int representing dimensionality of embedding vector
        :param units: int representing num of hidden units in RNN cell
        :param batch: int representing batch size
        Public Instances
        embedding: Keras Embedding layer converts words from vocab into an
                    embedding vector
        gru: keras GRU layer with units=units
            return full sequence of outputs
            return  last hidden state
            weights initialized with glorot_uniform
        F: Dense layer with vocab units
        """
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units, recurrent_initializer="glorot_uniform",
                                       return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Public Instance method
        :param x: tensor shape(batch, 1)
                  contains prev word in target sequence as an index of target vocabulary
        :param s_prev: tensor shape(batch, units)
                  contains prev decoder hidden state
        :param hidden_states: tensor shape(batch, imput_seq_len, units)
                  contains outputs of the encoder
        Note: concatenate context vector with x
        :return: y, s
            y: tensor shape(batch, vocab)
                contains output word sd s one hot vector in the target vocabulary
            s: tensor shape(batch, units)
                contains new decoder hidden state
        """
        x_embed = self.embedding(x)
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        context = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(context, initial_state=hidden_states[:, -1])
        y = self.F(Outputs)
        return y, s
