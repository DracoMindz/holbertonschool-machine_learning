#!/usr/bin/env python3
"""
Function has the trained agent play an episode
"""
import numpy  as np
import gym

def play(env, Q, max_steps=100):
    """
    PLay an episode
    :param env: FrozenLakeEnv instance
    :param Q: np.ndarray; contains Q-table
    :param max_steps:
    :return:
    """