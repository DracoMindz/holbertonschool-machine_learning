#!/usr/bin/env python3
"""
Function performs Q-learning
"""
import numpy as np
import gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


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

    rewards_all_episodes = []
    for episode in range(num_episodes):
        state = env.reset()

        done = False
        rewards_currentEpisode = 0

        # nested loop runs within each timestep
        for step in range(max_steps_per_episode):
            # exploration or exploitation, epsilon
            action = epsilon_greedy(Q, state, exploration_rate)

            newState, reward, done, info = env.step(action)
            # update Q-table
            if (done and reward == 0):
                reward = -1

            explDecayMax = (learning_rate * (reward + discount_rate *
                                             np.max(Q[newState, :])))
            Q[state, action] = Q[state, action] * (1 - learning_rate)
            Q[state, action] = Q[state, action] - explDecayMax
            state = newState
            rewards_currentEpisode += reward

            if done is True:
                break
        expDiff = (max_exploration_rate - min_exploration_rate)
        expDecay = np.exp(-exploration_decay_rate*episode)
        exploration_rate = min_exploration_rate + expDiff * expDecay

        # add current rewards to total rewards list
        rewards_all_episodes.append(rewards_currentEpisode)
    return Q, rewards_all_episodes
