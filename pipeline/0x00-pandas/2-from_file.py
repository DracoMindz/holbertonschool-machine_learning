#!/usr/bin/env python3
"""
Function loads from a file as a DataFrame
"""

import numpy as np
import pandas as pd


def from_file(filename, delimiter):
    """
    loads from file
    :param filename: the file to load from
    :param delimiter: the column separator
    :return: loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
