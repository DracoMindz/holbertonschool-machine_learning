#!/usr/bin/env python3
"""
Function creates the loss tensor and training op for the
generator
"""

import numpy as np
import tensorflow as tf

generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """
    creates the loss tensor and training for generator
    :param Z: tf.placeholder, contains: input for generator
    :param X: tf.placeholder, contains real input for discriminator
    Note: Generator: minimize negative modified minimax loss
                         be trained using Adam optimization
          Discriminator: not trained
          loss: gen loss
          train_op: training operation for the gen
    :return: loss, train_op
    """
    # models
    gen_Z = generator(Z)  # fake
    disc_x_fake = discriminator(gen_Z)

    # Generator Loss
    gen_loss = -tf.reduce_mean(tf.log(disc_x_fake))

    # variables
    genr_vars = [var for var im tf.trainable_variables()
                 if var.name.startswith("gen")]

    # Optimizer
    gen_train_op = tf.train.AdamOptimizer().minimize(gen_loss,
                                                     var_list(genr_vars))

    return gen_loss, gen_train_op
