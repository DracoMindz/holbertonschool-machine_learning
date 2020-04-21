#!/usr/bin/env python3
"""
function evaluates output of a neural network
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    X: numpy.ndarray input data to evaluate
    Y: numpy.ndarray one-hot labels for x
    save_path: location to load modelfrom
    Returns: network's prediction, accuracy, and loss
    """

    with tf.Session() as sess:
        save = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        dictFeed = {x: X, y: Y}

        out_prediction = sess.run(y_pred, dictFeed)
        out_accuracy = sess.run(accuracy, dictFeed)
        out_loss = sess.run(loss, dictFeed)

        return out_prediction, out_accuracy, out_loss
