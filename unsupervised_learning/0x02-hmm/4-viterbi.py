#!/usr/bin/env pythion3
"""
Function calculates most likely sequence of hidden
states for a hidden markov model
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates most likely sequence of hidden states
    :param Observation: np.ndarray, (T,) index of observation
    :param Emission: np.ndartray, (N, M) emission prob of a specific
                    observation given a hidden state
    :param Transition: np.ndarray, (N, N) transition probabilities
    :param Initial: np.ndarray, (N, 1) prob of starting in a
                    particular hidden state
            T: number of observations
            Emission[i, j]: prob of observing j given hidden state i
            N: number of hidden states
            M: number of all possible Observations
            Transition[i, j]: prob of transitioning from hidden
                            state i to j
            path: list of length T contains: most likely sequence of
                            hidden states
            P: prob of obtaining the path sequence
    :return: path, P, or None, None
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
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    T = Observation[0]
    N, M = Emission.shape
    N, N = Transition.shape
    vitAlg = np.empty((N, T))
    backTrak = np.empty((N, T))

    vitAlg[:, 0] = Initial.T * Emission[:, Observation[0]]
    backTrak[:, 0] = 0

    for i in range(1, T):
        vitAlg[:, i] = np.max(vitAlg[:, i - 1] * Transition.T *
                              Emission[np.newaxis, :, Observation[i]].T, 1)
        backTrak[:, i] = np.argmax(vitAlg[:, i - 1] * Transition.T, 1)

    for i in range(T):
        x = 0
    x[- 1] = np.argmax(vitAlg[:, 0])
    for i in reversed(range(1, T)):
        b = backTrak[x[i], i]
        x[i - 1] = int(b)

    P = np.amax(vitAlg, axis=0)   # P not float
    P = np.amin(P)
    return x, P
