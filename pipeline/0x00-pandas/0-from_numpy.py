#!/usr/fbin/env python3
"""
Function creates a data frame from a numpy array
"""

import numpy as np
import pandas as pd


def from_numpy(array):
    """
    creates a pd.DataFrame from a np.ndarray
    :param array: np.ndarray from which to create a pd.DataFrame
    :return: pd.DataFrame
    """
    df = pd.DataFrame(array, columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                      [:array.shape[1]])
    return df
