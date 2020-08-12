#!/usr/bin/env python3
"""
Function has the trained agent play an episode
"""
import numpy as np
import gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """
    PLay an episode
    :param env: FrozenLakeEnv instance
    :param Q: np.ndarray; contains Q-table
    :param max_steps: max num step in episode
    Note: Each state of the bnoard should be displayed via the console
    Note: should always exploit the Q-table
    :return: Total Rewards for the episode
    """
    for episode in range(1):
        state = env.reset()
        env.render()
        done = False
        for step in range(max_steps):
            env.render()
            action = np.argmax(Q[state, :])
            state, reward, done, info = env.step(action)
            if done:
                break
            # state = newState
    return reward
