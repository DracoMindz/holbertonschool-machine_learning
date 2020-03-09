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
        negfp = np.delete(confusion, m, 1)
        negfn = np.delete((negfp), m, 0)
        tNFP = np.delete(confusion, m, 0).sum()
        return negfn / tNFP
