#!/usr/bin/env python3
"""
Python script to train agent to play Atari's Breakout
"""
import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.processors import Processor
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from tensorflow.keras import layers


# rewards_all_episodes = []
class AtariProcessor(Processor):
    # Atari Processor class

    def process_Observation(selfself, observation):
        # observation (height, width, channel)
        assert observation.ndim == 3
        img = Image.fromarray(observation)

        # (resize, convert to grayscale)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)

        # storred in expereince memory
        return processed_observation.astype('unit8')

    def process_state_batch(selfself, batch):
        # Performing process here can use 'unit8' instead of 'float32'
        processed_batch = batch.astype('float32') / 255
        return processed_batch

    def process_reward(self, reward):
        # processing reward
        return np.clip(reward, -1, 1)


def create_q_model(actionNum, window):
    # create model define layers
    # inputs
    inputs = layers.Input(shape=(window, 84, 84))
    inputs_p = layers.Permute((2, 3, 1))(inputs)

    # convolutional layers
    layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs_p)
    layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = Flatten()(layer3)
    layer5 = Dense(512, activation="relu")(layer4)
    action = Dense(actionNum, activation="linear")(layer5)
    qModel = Model(inputs=inputs, outputs=action)
    return qModel


# model makes predictions for Q-values use for actions
modelPred = create_q_model()

# target model predicts future rewards
modelTarget = create_q_model()

if __name__ == '__main__':
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
                   processor=processor, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=1750000, log_interval=10000,
            visualize=False, verbose=2)

    # saving weights after training done
    dqn.save_weights('policy.h5', overwrite=True)
