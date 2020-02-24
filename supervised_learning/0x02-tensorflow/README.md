# 0x02. Tensorflow
Specialization - Machine Learning

***General***
---
What is tensorflow?
What is a session? graph?
What are tensors?
What are variables? constants? placeholders? How do you use them?
What are operations? How do you use them?
What are namespaces? How do you use them?
How to train a neural network in tensorflow
What is a checkpoint?
How to save/load a model with tensorflow
What is the graph collection?
How to add and get variables from the collection
---

***Optimize Tensorflow (Optional)***

In order to get full use of your computer’s hardware, you will need to build tensorflow from source.

###Tasks

0. Placeholders mandatory
Write the function def create_placeholders(nx, classes): that returns two placeholders, x and y, for the neural network

1. Layers mandatory
Write the function def create_layer(prev, n, activation)

2. Forward Propagation mandatory
Write the function def forward_prop(x, layer_sizes=[], activations=[]): that creates the forward propagation graph for the neural network

3. Accuracy mandatory
Write the function def calculate_accuracy(y, y_pred): that calculates the accuracy of a prediction

4. Loss mandatory
Write the function def calculate_loss(y, y_pred): that calculates the softmax cross-entropy loss of a prediction

5. Train_Op mandatory
Write the function def create_train_op(loss, alpha): that creates the training operation for the network

6. Train mandatory
Write the function def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"): that builds, trains, and saves a neural network classifier

7. Evaluate mandatory
Write the function def evaluate(X, Y, save_path): that evaluates the output of a neural network

