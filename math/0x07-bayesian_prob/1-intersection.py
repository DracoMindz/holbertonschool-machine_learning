#!/usr/bin/env python3
"""
calculates the intersection of obtaining this data
with the various hypothetical probabilities
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this data
    with the various hypothetical probabilities
    :param x: num of patients that develope side effects
    :param n: tot num of patients observed
    :param P: 1D np.ndarray containing prior beliefs
    :param Pr: 1D np.ndarray containing prior beliefs
    :return: 1D np.ndarray containing intersection of
    obtaining x and n  with each probability in P.
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
    likelihood = factNX * (P ** x) * ((1 - P) ** (n - x))
    return likelihood * Pr
