#!/usr/bin/env python3
"""
Class Decoder
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    class c reates the decoder for a tranformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 xmax_seq_len, drop_rate=0.1):
        """
        class constructor
        param:: N: number of blocks in the encoder
        param:: dm: dimensionality of the model
        param:: h: number of heads
        param:: hidden: number of hidden units in the fully connected layer
        param:: target_vocab: size of the target vocabulary
        param:: max_seq_len: maximum sequence length possible
        param:: drop_rate: dropout rate
        Public Instances
        N: nummber of blocks in the encoder
        dm: dimensionality of the model
        embedding: embedding layer for the targets
        positional_encoding: numpy.ndarray of shape
                    (max_seq_len, dm) containing the
                    positional encodings
        blocks: list of length N containing all of the DecoderBlock‘s
        dropout: dropout layer, to be applied to the positional encodings
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for m in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        x: tensor of shape (batch, target_seq_len, dm)
                            containing the input to the decoder
        encoder_output: tensor of shape (batch, input_seq_len, dm)
                        containing the output of the encoder
        training: boolean to determine if the model is training
        look_ahead_mask: the mask to be applied to the first
                        multi head attention layer
        padding_mask: the mask to be applied to the second multi
                        head attention layer
        Returns: tensor of shape (batch, target_seq_len, dm)
                 containing the decoder output
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for idx in range(self.N):
            dec_output = self.blocks[idx](x, encoder_output, training,
                                          look_ahead_mask, padding_mask)
        return dec_output
