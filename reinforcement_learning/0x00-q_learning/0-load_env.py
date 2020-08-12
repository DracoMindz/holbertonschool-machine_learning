#!/usr/bin/env python3
"""
Function loads pre-made environment from OpenAI's gym
"""

import numpy
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads environment from OpenAI's gym
    :param desc: None or list;
            contains: decsription of map to load for the envoronment
    :param map_name: None or string
                    contains: pre-made map to load
    :param is_slippery: boolean; determines if the ice is slippery
    Note: If both desc and map_name are None envoronment will load
            random generated 8x8 map
    :return: the environment
    """

    env = gym.make("FrozenLake-v0", desc=desc,
                   map_name=map_name, is_slippery=is_slippery)
    return env
