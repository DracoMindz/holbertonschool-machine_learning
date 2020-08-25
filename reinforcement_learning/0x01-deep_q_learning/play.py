#!/usr/bin/env python3
"""
write a script that can display a game played by the agent
"""

import numpy as np
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreed$yQPolicy
from rl.processors import Processor

create_q_model = __import__('train').create_q_model
Atari_Processor = __import__('train').Atari_Processor


if __name__ == '__main__':
    """PLay an episode of the game"""
    windowLen = 4
    # get the environment and the nuber of actions
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    model = create_q_model(num_actions, windowLen)
    memory = SequentialMemory(limit=1000000, window_length=windowLen)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1,
                                  value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # training the model
    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   policy=policy, memory=memory,
                   processor=processor)
    dqn.cmopile(Adam(lr=.00025), metrics=['mae'])

    # loading weights after training is done
    dqn.load_weights('policy.h5')

    # Evaluating algorithm
    dqn.test(env, nb_episodes=10, visualize=True)
