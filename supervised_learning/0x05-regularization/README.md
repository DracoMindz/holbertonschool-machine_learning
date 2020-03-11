# 0x05. Regularization
Machine Learning ― Supervised Learning

![alt text}](https://banner2.cleanpng.com/20180411/uce/kisspng-deep-learning-machine-learning-artificial-neural-n-networking-5ace71b40f6c63.2083189615234789640632.jpg)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### ***General***
---
```
What is regularization? What is its purpose?

What is are L1 and L2 regularization? What is the difference between the two methods?

What is dropout?

What is early stopping?

What is data augmentation?

How do you implement the above regularization methods in Numpy? Tensorflow?

What are the pros and cons of the above regularization methods?
```
---

## ***Tasks***

***0. L2 Regularization Cost***

Write a function def l2_reg_cost(cost, lambtha, weights, L, m): that calculates the cost of a neural network with L2 regularization

***1. Gradient Descent with L2 Regularization***

Write a function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): that updates the weights and biases of a neural network using gradient descent with L2 regularization

***2. L2 Regularization Cost***

Write the function def l2_reg_cost(cost): that calculates the cost of a neural network with L2 regularization

***3***. Create a Layer with L2 Regularization***

Write a function def l2_reg_create_layer(prev, n, activation, lambtha): that creates a tensorflow layer that includes L2 regularization

***4. Forward Propagation with Dropout***

Write a function def dropout_forward_prop(X, weights, L, keep_prob): that conducts forward propagation using Dropout

***5. Gradient Descent with Dropout***

Write a function def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L): that updates the weights of a neural network with Dropout regularization using gradient descent


***6. Create a Layer with Dropout***

Write a function def dropout_create_layer(prev, n, activation, keep_prob): that creates a layer of a neural network using dropout

***7. Early Stopping***

Write the function def early_stopping(cost, opt_cost, threshold, patience, count): that determines if you should stop gradient descent early