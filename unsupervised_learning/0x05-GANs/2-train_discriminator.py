#!/usr/bin/env python3
"""
Function that creates the loss tensor and training op for the discriminator
"""

import numpy as np
import tensorflow as tf

generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """
    creates loss tensor and training op
    :param Z: tf.placeholder, contains: input for generator
    :param X: tf.placeholder, contains real input for discriminator
    Note: Discriminator: minimize negative minimax loss
                         be trained using Adam optimization
          Generator: not trained
          loss: discrminator loss
          train_op: training operation for the discriminator
    :return: loss, train_op
    """
    # fake and reakl
    gen_Z = generator(Z)  # fake
    disc_real = discriminator(X)  # real image
    disc_x_fake = discriminator(gen_Z)

    # loss
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1 - gen_Z))

    # trainable variables
    dis_vars = [vars for var in tf.trainable_variables()
                if var.name.startswith("disc")]

    # Optimize
    disc_train_op = tf.train.AdamOptimizer().minimize(disc_loss,
                                                      var_list=dis_vars)

    return (disc_loss, disc_train_op)
