#!/usr/bin/env python3
"""
Function that trains a GAN
"""

import numpy as np
import  tensoflow as tf


train_generator = __import__('2-train_generator').train_generator
train_discriminator = __import__('3-train_discriminator').train_discriminator
sample_Z = __import__('4-sample_Z').sample_Z


def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):
    """
    Trains a GANs
    :param X: np.ndarray, shape (m, 784}
    :param epochs: num of epochs each network is trained for
    :param batch_size: batch sized used during the training
    :param Z_dim: num of dimensions for the randomly generated input
    :param save_path: path to save the trained generator
    NOte: create tf.placeholger for Z
          add Z to the graph's collection
    Note: Discriminator and Generator: training should be
           altered after one epoch
    :return:
    """
    for ep in range(epochs):
        for bat in X:





