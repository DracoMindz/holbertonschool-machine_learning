#!/usr/bin/env python3
"""
Class Encoder
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class Constructor
        param:: :N: number of blocks in the encoder
        param:: dm: dimensionality of the model
        param:: h: number of heads
        param:: hidden: number of hidden units in the fully connected layer
        param:: input_vocab: size of the input vocabulary
        param:: max_seq_len: maximum sequence length possible
        param:: drop_rate: dropout rate
        Public Instances
        N - number of blocks in the encoder
        dm - dimensionality of the model
        embedding - the embedding layer for the inputs
        positional_encoding - a numpy.ndarray of
                    shape (max_seq_len, dm) containing the
                    positional encodings
        blocks - a list of length N containing all of the EncoderBlock‘s
        dropout - the dropout layer, to be applied to the positional encodings

        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for m in range(self.N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Public Instance Method
        create a decoder for a transformer
        param:: x: tensor shape(batch, input_seq_len, dm)
                containing the input to the encoder
        param:: training: boolean to determine if the model is training
        partam:: mask: mask to be applied for multi head attention
        Returns: tensor shape(batch, input_seq_len, dm)
                containing the encoder output
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        enc_output = self.dropout(x, training=training)
        for idx in range(self.N):
            enc_output = self.blocks[idx](enc_output, training, mask)
        return enc_output
