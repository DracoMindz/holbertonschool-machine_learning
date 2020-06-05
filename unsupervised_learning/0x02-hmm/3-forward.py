#!/usr/bin/env/pyhton3
"""
Function performs forward algorithm for hidden markov
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    forward algorithm for a hiddem markov model
    :param Observation: np.ndarray, (T,) contains: index of observation
    :param Emission: np.ndarray, (N,M),
                    contains: emission prob of observation given state
    :param Transition: 2D np.ndarray, (N,N) contains: transition probs
    :param Initial: np.ndarray, (N,1), contains: prob of starting a
                    particular hidden state
        T: num observations
        N: num of hidden states
        M: num of all possible observations
        P: lilklihood of the observations given fmodel
        F: np.ndarray, (N,T), contains: forward path probabilities
        F[i, j]: probality  of being hidden state i at time j
                given previous observations
        :return: P, F, or None, None
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Observation.shape[0] < 1:
        return None, None
    if ((Transition.shape[0] != Transition.shape[1] != Emission.shape[0]
         != Initial.shape[0] or Initial.shape[1] != 1)):
        return None, None
    T = Observation[0]
    N, M = Emission.shape
    N, N = Transition.shape
    F = np.zeros((N, T))
    #  second position in  return second position in []
    #  F = alpha
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, (t - 1)] * Transition[:, j]
                             * Emission[j, Observation[t]])
    P = float(np.sum(F[:, -1:]))
    return P, F
