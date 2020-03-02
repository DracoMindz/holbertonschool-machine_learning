#!/usr/bin/env python3
"""function builds, trains, and saves a neural network classifier"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):

    """
    X_train is numpy.ndarray containing training input data
    Y_train is  numpy.ndarray containing training labels
    X_valid is  numpy.ndarray containing validation input data
    Y_valid is  numpy.ndarray containing validation labels
    layer_sizes is list containing num nodes in layer of network
    actications is list containing activation functions for each layer of N
    alpha is learning rate
    iterations is num iterations to train over
    save_path designates where to save model
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train = create_train_op(loss, alpha)
    session = tf.Session()
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collectino('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train', train)
    tf.Graph.get_collection(x, y, y_pred, loss, accuracy, train)
    z = tf.global_variables_initializer()
    session.run(z)

    for i in range(0, iterations):
        if (i == 0) or (i % 100):
            print('After {} iterations:'.format(i))
            tLoss, taccuracy = session.run((loss, accuracy),
                                           feed_dict={x: X_train,
                                                      y: Y_train})
            print('\tTraining Cost:'.format(tLoss))
            print('\tTraining Accuracy:'.format(taccuracy))
            vLoss, vAccuracy = session.run((loss, accuracy),
                                           feed_dict={x: X_valid,
                                                      y: Y_valid})
            print('\tTraining Cost:'.format(vLoss))
            print('\tTraining Accuracy:'.format(vAccuracy))
        session.run(train, feed_dict={x: X_train, y: Y_train})
    return tf.train.Saver.save(session, save_path)
