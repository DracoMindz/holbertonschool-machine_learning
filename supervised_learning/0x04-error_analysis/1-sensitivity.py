#!/usr/bin/env python3
"""
calculates sensitivity for each class in a confusion matrix
"""


def sensitivity(confusion):
    """
    confusion: confusion numpy.ndarray of shape (classes, classes)
    classes: num of classes
    """
    sensitivity_arr = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    return sensitivity_arr
