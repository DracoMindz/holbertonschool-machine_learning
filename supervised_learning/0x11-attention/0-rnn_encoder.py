#!/usr/bin/evn python3
"""
Class RNNEncoder
inherits from tensorflow.keras.layers.Layer to encode
machine translators
"""
import numpy
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNncoder Class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
        :param vocab: integer represents size of input vocab
        :param embedding: int represents dimensionality of embedding vector
        :param units: int represents num of hidden units in the RNN cell
        :param batch: int represents batch size
        Sets Public Instance attributes:
        batch: batch size
        units: num of hidden unit in the RNN
        embedding: keras embedding layer that converts words from vocab
                    into embedding vector
        gru: keras GRU layer with units units
            return full sequence of outputs
            return hidden state
            recurrent weights intialized with glorot_uniform
        """
        self.batch =  batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units, recurrent_initializer="glorot_uniform",
                                       return_sequences=True, return_state=True)


    def initialize_hidden_state(self):
        """
        Public instance method
        Initiailizes the hidden states for RNN cell to a tensor of zeros
        :return: a tensor shape(batch, units)
                contains: initialized hidden states
        """
        rnnten = tf.zeros(self.batch, self.units)
        return rnnten

    def call(self, x, initial):
        """
        Public instanc method
        :param x: tensor shape(batch, input_seq_len, units)
                containing input to the encoder layer as word indicies
                within the vocab
        :param initial: tensor shape(batch, units)
                        contains: initial hidden state
        :return: outputs, hidden
        Note: outputs: tensor shape(batch, input_seq_len, units)
                        contains outputs of the encoder
             hidden: tensor of shape(batch, units)
                        contains last hidden state of the encounter
        """
        outputs, hidden = self.gru(self.embedding(x), initial_state=initial)
        return outputs, hidden
