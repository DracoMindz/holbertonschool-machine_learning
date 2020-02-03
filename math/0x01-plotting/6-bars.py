#!/usr/bin/env python3
"""Function creates a Bar graph"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
# the bars and labels
appleBar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[0], width=.5,
                   color='red', label='apples')
bananaBar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[1], width=.5,
                    color='yellow', label='bananas', bottom=fruit[0])
orangeBar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[2], width=.5,
                    color='#ff8000', label='orange',
                    bottom=fruit[0] + fruit[1])
peachBar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[3], width=.5,
                   color='#ffe5b4', label='peach',
                   bottom=fruit[0] + fruit[1] + fruit[2])
plt.legend()

# design of the graph
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.ylim(0, 80, 10)
plt.show()
