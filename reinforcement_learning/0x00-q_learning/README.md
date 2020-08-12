# 0x00 Q-learning
Specialization Machine Learning - Reinforcement Learning

## Learning Objectives

    What is a Markov Decision Process?
    What is an environment?
    What is an agent?
    What is a state?
    What is a policy function?
    What is a value function? a state-value function? an action-value function?
    What is a discount factor?
    What is the Bellman equation?
    What is epsilon greedy?
    What is Q-learning?

## Installations for this Project
```
pip install --user gym

pip3 install --user gym (for python3)
```

### Tasks

**0. Load the Environment**

Write a function def load_frozen_lake(desc=None, map_name=None,
is_slippery=False): that loads the pre-made FrozenLakeEnv evnironment
from OpenAI’s gym.
___
**1. Initialize Q-table**
Write a function def q_init(env): that initializes the Q-table.

___
**2. Epsilon Greedy**

Write a function def epsilon_greedy(Q, state, epsilon) that uses
epsilon-greedy to determine the next action.

___
**3. Q-learning**

Write the function def train(env, Q, episodes=5000, max_steps=100,
alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
epsilon_decay=0.05): that performs Q-learning.

___
**4. Play**

Write a function def play(env, Q, max_steps=100): that has the
trained agent play an episode.

___






