#!/usr/bin/env python3
"""
Function performs PCA color augmentation on image
"""

import tensorflow as tf


def pca_color(image, alphas):
    """
    PCA color augmentation on image
    :param image: 3D tf.Tensor
    :param alphas: tuple of len 3
                    contains amount channel should change
    :return: augmented image
    """

    # creates copy of image
    # normalize image
    imageX = tf.image.per_image_standardization(image)

    # calcuate eigen values
    eigen_values, eigen_vectors = tf.linalg.eig(imageX, name=None)
    # use eigen values to rotate data
    imageEigen = tf.tensordot(tf.transpose(eigen_vectors),
                              tf.transpose(imageX),
                              axes=1)

    pca_color_image = np.maximum(np.minimum
                                 (imageEigen + alphas, 255),
                                 0).astype('uint8')

    return pca_color_image
