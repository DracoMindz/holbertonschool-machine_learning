#!/usr/bin/env python 3
"""
Function creates a sparse autoencoder
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras as K


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder
    :param input_dims: int, containing dimensions of model input
    :param hidden_layers: list, contains num of nodes for layer
    :param latent_dims: int, contains dimensions of latent space
    :param lambtha: regularization, used for L1 Reg on encoded input
        note: hiden layers should be reversed for the decoder
        note: autoencoder model should be compiled using adam
                and binary cross-entropy loss
        note: all layers use a relu activation except the last layer
        note: last lahyer in the decoder uses sigmoid
    :return: encoder, decoder, auto
    """
    # inputs
    encoderInput = K.Input(shape=(input_dims,))
    decoderInput = K.Input(shape=(latent_dims,))

    # first output layers
    en_output_1 = K.layers.Dense(hidden_layers[0],
                                 activation='relu')(encoderInput)
    de_output_1 = K.layers.Dense(hidden_layers[-1],
                                 activation='relu')(decoderInput)

    # Encoder
    for idx in range(1, len(hidden_layers)):
        # second output layer
        en_output_2 = K.layers.Dense(hidden_layers[idx],
                                     activation='relu',
                                     activity_regularizer=K.regularizers.l1
                                     (lambtha))(en_output_1)
        encoderOutput = K.layers.Dense(latent_dims,
                                       activation='relu')(en_output_2)

    # encoder Model
    encoder = K.models.Model(inputs=encoderInput, outputs=encoderOutput)

    # Decoder
    for idx in range(len(hidden_layers) - 2, -1, -1):
        # second output layer
        de_output_2 = K.layers.Dense(hidden_layers[idx],
                                     activation='relu')(de_output_1)
        decoderOutput = K.layers.Dense(latent_dims,
                                       activation='sigmoid')(de_output_2)

    # decoder Model
    decoder = K.models.Model(inputs=decoderInput, outputs=decoderOutput)

    # Autoencoder
    autoInput = K.Input(shape=input_dims, )
    encoderAuto = encoder(autoInput)
    decoderAuto = decoder(encoderAuto)
    auto = K.Model(inputs=autoInput, outputs=decoderAuto)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
