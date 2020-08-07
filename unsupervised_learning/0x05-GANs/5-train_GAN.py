#!/usr/bin/env python3
"""
Function that trains a GAN
"""

import numpy as np
import tensorflow as tf


train_generator = __import__('3-train_generator').train_generator
train_discriminator = __import__('2-train_discriminator').train_discriminator
sample_Z = __import__('4-sample_Z').sample_Z


def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):
    """
    Trains a GANs
    :param X: np.ndarray, shape (m, 784), contains:Data input
    :param epochs: num of epochs each network is trained for
    :param batch_size: batch sized used during the training
    :param Z_dim: num of dimensions for the randomly generated input
    :param save_path: path to save the trained generator
    NOte: create tf.placeholger for Z
          add Z to the graph's collection
          m is num data samples

    Note: Discriminator and Generator: training should be
           altered after one epoch
    :return:
    """
    if not save_path:
        return exit(1)
    else:
        fileLocation = np.pathlib.Path(save_path)

    for ep in range(epochs):
        for batch in X:
            noise = tf.random.normal(shape=[batch_size, Z_dim])
            genImages = train_generator(noise)
            X_realFake = tf.concat([genImages, batch], axis=0)
            y_1 = tf.constant([[0]] * batch_size + [[1]] * batch_size)
            train_discriminator.trainable = True
            train_discriminator.train_on_batch(X_realFake, y_1)
            noise = tf.random.normal(shape=[batch_size, Z_dim])
            y_2 = tf.constant([[1]] * batch_size)
            train_discriminator.trainable = False
            train_generator.train_on_batch(noise, y_2)

    generator = train_gan(X, batch_size=batch_size, Z_dim=Z_dim)
    generator = generator(fileLocation)
