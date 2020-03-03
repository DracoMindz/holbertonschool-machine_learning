#!/usr/bin/env python3
"""function calculates weighed moving average of a data set"""

import numpy as np
import tensorflow as tf


def moving_average(data, beta):
    """
    data: list of data to calculate the moving average of
    beta: weight used for the moving average
    """

    ewa = [0]
    bias_corrected = []

    for k_index, p_data in enumerate(data):
        ewa.append(beta * ewa[k_index] + (1 - beta) * p_data)
        bias_corrected.append(ewa[k_index + 1] / (1 - beta ** (k_index + 1)))
    return bias_corrected
