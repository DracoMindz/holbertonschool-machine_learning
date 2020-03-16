# 0x06. Keras
Specializations - Machine Learning ― Supervised Learning

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### ***General***
___
```
What is Keras?
What is a model?
How to instantiate a model (2 ways)
How to build a layer
How to add regularization to a layer
How to add dropout to a layer
How to add batch normalization
How to compile a model
How to optimize a model
How to fit a model
How to use validation data
How to perform early stopping
How to measure accuracy
How to evaluate a model
How to make a prediction with a model
How to access the weights/outputs of a model
What is HDF5?
How to save and load a model’s weights, a model’s configuration, and the entire model
```
___

### Tasks

***0. Sequential mandatory***

Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library

***1. Input mandatory***

Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library

***2. Optimize mandatory***

Write a function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics

***3. One Hot mandatory***

Write a function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix

***4. Train mandatory***

Write a function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descent

***5. Validate mandatory***

Based on 4-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data

***6. Early Stopping mandatory***

Based on 5-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping

***7. Learning Rate Decay mandatory***

Based on 6-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay

***8. Save Only the Best mandatory***

Based on 7-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model

***9. Save and Load Model mandatory***

Write the following functions:
def save_model(network, filename): saves an entire model

***10. Save and Load Weights ***

***11. Save and Load Configuration***

***12. Test ***

***13. Predict mandatory***

Write a function def predict(network, data, verbose=False): that makes a prediction using a neural network
