#!/usr/bin/env python3
"""
write a script that can display a game played by the agent
"""

import numpy as np
import gym


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
    for episode in range(50):
        state = env.reset()
        env.render()
        done = False
        for step in range(max_steps):
            env.render()
            action = np.argmax(Q[state, :])
            # state, reward, done, info = env.step(action)
            done = False
            while not done:
                state, reward, done, _ = env.step(env.action)
                env.render()
    return reward
