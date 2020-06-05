#!/usr/bin/env python3
"""
Function performs the backward algorithm for a hidden markov model
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Backward algorithm for a hidden markov model
    :param Observation: np.ndarray, (T,) index of observation
    :param Emission: np.ndarray, (N,N) emission prob of specific
                observation
    :param Transition: np.ndarray, (N,N) tranisition probability
    :param Initial: np.ndarray, (N, 1) probability of startinng in
                    a particular hidden state
        T: num  of observations
        Emission[i,j]: probability of observing j given hidden state i
        N: number hidden states
        M: number of all possible observations
        Transition[i,j]: probability of tranisitioning from the hidden

    :return: P, B, or None
    """