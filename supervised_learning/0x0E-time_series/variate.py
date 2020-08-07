#!/usr/bin/python3
"""
Forcast
"""
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def multivariate_data(dataset, target, startIdx, endIdx,
                      historySize, targetSize, step,
                      single_step=False):
    """
    multivariate data
    :return: data, labels
    """
    data = []
    labels = []
    #
    startIdx = startIdx + historySize
    if endIdx is None:
        endIdx = len(dataset) - targetSize

    for i in range(startIdx, endIdx):
        indices = range(i-historySize, i, step)

        data.append(dataset[indices])
    if single_step:
        labels.append(target[i + targetSize])
    else:
        labels.append(target[i:i + targetSize])

    return np.array(data), np.array(labels)


def univariate_data(dataset, startIdx, endIdx, historySize, targetSize):
    """
    univariate data
    :return: data, labels
    """
    data = []
    labels = []

    startIdx = startIdx + historySize
    if endIdx is None:
        endIdx = len(dataset) - targetSize

    for i in range(startIdx, endIdx):
        indices = range(i-historySize, i)

        data.append(np.reshape(dataset[indicies], (historySize, 1)))
        labels.append(dataset[i+targetSize])

    return np.array(data), np.array(labels)


def univariate():
    """
    data split
    """
    pastHistory = 24
    futureTarget = 0

    xTrain, yTrain = univariate_data(dataset, 0, TRAIN_SPLIT, pastHistory,
                                     futureTarget)
    xTest, yTest = univariate_data(dataset, TRAIN_SPLIT, pastHistory,
                                   futureTarget)
