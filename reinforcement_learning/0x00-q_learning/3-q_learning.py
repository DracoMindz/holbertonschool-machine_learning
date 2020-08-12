#!/usr/bin/env python3
"""
Function performs Q-learning
"""
import numpy as np
import gym


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs Q-learning
    :param env: FrozenLakeEnv instance
    :param Q: np.ndarray, Q-table
    :param episodes: num of epiosodes to train over
    :param max_steps: max num steps per episode
    :param alpha: learning rate
    :param gamma: discount rate
    :param epsilon: initial threshold for epsilon greedy
    :param min_epsilon: min val of epsilon decay
    :param epsilon_decay: decay rate for updating tew episodes
    Note: when agen falls ogin a hole reward is -1
    :return: Q, total_rewards
        total_rewards: list of rewards per episode
    """
    num_episodes = episodes
    max_steps_per_episode = max_steps
    learning_rate = alpha
    discount_rate = gamma
    exploration_rate = epsilon
    max_exploration_rate = epsilon
    min_exploration_rate = min_epsilon
    exploration_decay_rate = epsilon_decay
