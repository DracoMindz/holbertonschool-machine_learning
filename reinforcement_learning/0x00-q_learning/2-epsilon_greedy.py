#!/usr/bin/env python3
"""
Function uses epsilon-greedy to determine next action
"""

import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """
    determin next action
    :param Q: np.ndarray, contains: q-table
    :param state: current state
    :param epsilon: use for calculations
    :return: next action
    """
    # determine wether to explore or exploit
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(Q.shape[1])
