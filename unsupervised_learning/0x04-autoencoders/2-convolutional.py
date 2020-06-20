#!/usr/bin/env python 3
"""
Function creates a convolutional autoencoder
"""

import tensorflow as tf
from tensorflow import keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    creates convolutional autoencoder
    :param input_dims: tuple of ints, contains demensions
                        of model input
    :param filters: list, contains num filters for each
                    convolutional layer in encoder
    :param latent_dims: tuple of ints, contains dimensions
                    of latent space representation
        Encoder Notes
        note: filters should be reversed from decoder
              use a kernel size (3,3) with same padding
              use relu activation
              followed by max pooling size (2,2)
        Decoder Notes
        note: each convolution in the decoder ( except last two)
                filter size (3,3) with same padding
                relu activation
                followed by upsampling size (2,2)
        note: second to last layer use valid padding
        note: last layer num filters = num channel in input_dims
              sigmoid activation
              no upsampling
        Autoencoder Notes
        note: autoencoder model compiled using Adam optimization
              and binary cross-entrophy loss
    :return: encoder, decoder, auto
    """
    encInput = K.layers.Input(shape=input_dims)
    decInput = K.layers.Input(shape=latent_dims)

    # Encoded part of model
    en_layer = K.layers.Conv2D(filters=filters[0], kernel_size=3,
                               activation='relu', padding='same')(encInput)
    encoded = K.layers.MaxPooling2D(pool_size=(2, 2))(en_layer)

    for f in range(1, len(filters)):
        encoded_fLay = K.layers.Conv2D(filters=filters[f],
                                       kernel_size=3, activation='relu',
                                       padding='same')(encoded)
        encoded_fPool = K.layers.MaxPooling2D(pool_size=(2, 2))(encoded_fLay)
    encoded_Output = K.layers.Reshape(latent_dims)(encoded_fPool)

    # Decoded part of model
    de_layer = K.layers.Conv2D(filters=filters[-1],
                               kernel_size=3, activation='relu',
                               padding='same')(decInput)
    decoded = K.layers.UpSampling2D(size=(2, 2))(de_layer)

    for f in range(len(filters)-2, 2, -1):
        decoded_fLay = K.layers.Conv2D(filters=filters[f],
                                       kernel_size=3, activation='relu',
                                       padding='same')(decoded)
        decoded_fSam = K.layers.UpSampling2D(size=(2, 2))(decoded_fLay)

    # Second to last layer and last layer, opposite direction from encoder
    decoded_2Last = K.layers.Conv2D(filters=filters[1],
                                    kernel_size=3, activation='relu',
                                    padding='valid')(decoded_fSam)
    decoded_fSam2 = K.layers.UpSampling2D(size=(2, 2))(decoded_2Last)
    decoded_Output = K.layers.Conv2D(filters=filters[0],
                                     kernel_size=3, activation='sigmoid',
                                     padding='same')(decoded_fSam2)

    # Encoder Model and Decoder Model
    encoder = K.models.Model(inputs=encInput, outputs=encoded_Output)
    decoder = K.models.Model(input=decInput, output=decoded_Output)

    # Autoencoder use encoder and decoder models
    autoInput = K.Input(shape=input_dems)
    auto_encOutput = encoder(autoInput)
    autoOutput = decoder(auto_encOutput)

    # Autoencoder Model
    auto = K.model.Model(inputs=autoInput, outputs=autoOutput)
    auto.K.model.compile(optimizer='Adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
