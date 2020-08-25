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
    
```   
Chellenges: There were challenges regarding the use of the keras-rl library.
It was necessary to install the rl2 library as opposed to the rl library. 
This diminished conflicts.  Error occurred regarding the version of tensorflow
being used.  Tensorflow 2.2 is the minimum requirement.  I was able to upgrade
to tensorflow 2.3.  If you have python and python3 installed before running the 
code you may need to install tensorflow 2.3 (or 2.2 at the minimum) as a "pip 
install" to and as "pip3 install" to avert conflicts.  

There is a __len__ conflict that occurrs when using Colabs suggesting to use tensorflow 1.14 
to avoid the conflict presented by keras-rl.
```

   
   