#!/usr/bin/env python3
"""
Function creates a variational autoencoder
"""

import tensorflow as tf
from tensorflow import keras as K


def take_sample(probs):
    """
    creates sample z
    :param inputs: tuple
    :return: sample
    """
    z_mean, z_log_var = probs
    eps = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]))
    sample = z_mean + (z_log_var) * eps
    return sample


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder
    :param input_dims: int, contains dimensions of model input
    :param hidden_layers: list, contains num nodes in hidden layer
                            in encoder
    :param latent_dims: int, contains dimensions fot latent space
    Notes: hidden layer shoulb be reverse for the decoder
            autoencoder model compiled with Adam optimization
                and binary cross-entropy loss
            All layers use relu activation
            mean and log layer in encoder use None
            last layer in decoder use sigmoid
    :return: encoder, decoder, auto
    """
    # Encoded part of model
    encInput = K.Input(shape=input_shape)
    encoded = K.layers.Dense(hidden_layers, activation='relu')(encInput)

    for i in range(1, len(hidden_layers)):
        encoded_hLay = K.layers.Dense(hidden_layers[i],
                                      activation='relu')(encoded)
        z_mean = K.layers.Dense(latent_dim,
                                activation+'None')(encoded_hLay)
        z_log_var = K.layers.Dense(latent_dim,
                                   activation='None')(encoded_hLay)
        z = take_sample()([z_mean, z_log_var])
    encoder = K.models.Model(inputs=encInput, outputs=[z_mean, z_log_var, z])
    encoder.K.summary()

    # Decoded part of Model

    decInput = K.Input(shape=latent_dims)

    for i in range(len(hidden_layer)-1, 1, -1):
        decoded_hLay = K.layers.Dense(hidden_layers[i],
                                      activation='relu')(decInput)
    decoded = K.layers.Dense(hidden_layers[0],
                             activation='sigmoid')(decoded_hLay)
    decoder = K.models.Model(latent_dim, decoded)
    decoder.summary()

    # VAE
    vaeInput = K.Input(shape=input_dims)
    vaeEncoder = encoder(vaeInput)
    vaeOutput = decoder(vaeEncoder)
    auto = K.Model(inputs=vaeInputs, outputs=vaeOutput)
    auto.K.model compile(optimizer='Adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
