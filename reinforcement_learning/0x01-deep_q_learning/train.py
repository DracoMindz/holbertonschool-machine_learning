#!/usr/bin/env python3
"""
Python script to train agent to play Atari's Breakout
"""
import gym
import numpy
import tensorflow as tf
import keras
import keras-rl
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Layer, Dense

from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl import SequentialMemory, DQNAgent

rewards_all_episodes = []


def load_Atari_env(desc=None, map_name=None, is_slippery=False):
    # Create game environment
    env = gym.make('Breakout-v0',
                   dec=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    # returns to starting game frame
    frame = env.reset()
    env.render()
    return env


def to_grayscale(img):
    # convert to gray
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    # down sample image
    return img[::2, ::2]


def processing(img):
    # process image
    return to_grayscale(downsample(img))


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    num_episodes = episodes
    max_steps_per_episode = max_steps
    learning_rate = alpha
    discount_rate = gamma
    exploration_rate = epsilon
    max_exploration_rate = epsilon
    min_exploration_rate = min_epsilon
    exploration_decay_rate = epsilon_decay
