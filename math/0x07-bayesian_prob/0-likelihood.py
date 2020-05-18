#!/usr/bin/env python3
"""
function calculats likelihood of obtaining data
given various hypothetical probabilities of developing
severe side effects
"""

import numpy as np


def likelihood(x, n, P):
    """calculats likelihood
    :param x: num patients that develop
    :param n: total number of patients observed
    :param p: np.ndarray, contains hypothetical probabilities
    Return: numpy.ndarray: containing likelihood of obtaining data
            x and n, for each probability in P
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that"
                         "is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    for val in P:
        if val > 1 or val < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
    factor = np.math.factorial
    factNX = (factor(n) / (factor(x) * factor(n - x)))
    return factNX * (P**x) * ((1 - P)**(n - x))
