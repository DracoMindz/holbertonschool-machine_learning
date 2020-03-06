#!/usr/bin/env python3
"""
creates a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    labels: one-hot numpy.ndarray shape (m, classes)
            containins correct labels each data point
    logits: one-hot numpy.ndarray shape (m, classes)
            containing the predicted labels
    m: num data points
    classes: numb classes
    """
    return np.dot(labels.T, logits)
