#!/usr/bin/env python3
"""
calculates the specificity each class in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    classes is the number of classes
    """
    for m in range(confusion.shape[0]):
        trueNeg = np.asarray(np.delete(np.delete(confusion, m, 0)), m, 1).sum()
        falsePos = np.sum(confusion, axis=0) - np.diagonal(confusion)
        specificity = trueNeg / (trueNeg + falsePos)
        return specificity
