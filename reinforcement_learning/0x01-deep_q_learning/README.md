# 0x01 Deep Q-Learning
Specialization- Machine Learning - Reinforcement Learning

## Learning Objectives

```
    What is Deep Q-learning?
    What is the policy network?
    What is replay memory?
    What is the target network?
    Why must we utilize two separate networks during training?
    What is keras-rl? How do you use it?
```
## Requirements

   **Installations:**
   ___
   pip install --user keras-rl
   ___

   Make sure the following is already installed.
   ___
   pip install --user keras==2.2.5
   pip install --user Pillow
   pip install --user h5py
   ___

   The Goal of this project is to create and agent that can play a game.

### Tasks

    **0. Breakout**
    Write a python script, train.py, that utilizes keras. keras-rl. and
    gym to train an agent that can play Atari's Breakout.

    Write a python script, play.py, that can display a game played
    by the agent trained by train.py.

    **The Agent should use:**
    keras-rl DQNAgent
    keras-rl SequentialMemory
    keras-rl EpsGreedyQPolicy
    

   
   