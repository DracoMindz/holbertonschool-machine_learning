# 0x03. Hyperparameter Tuning
Specializations - Machine Learning ― Unsupervised Learning

## Learning Objectives

```
What is Hyperparameter Tuning?
What is random search? grid search?
What is a Gaussian Process?
What is a mean function?
What is a Kernel function?
What is Gaussian Process Regression/Kriging?
What is Bayesian Optimization?
What is an Acquisition function?
What is Expected Improvement?
What is Knowledge Gradient?
What is Entropy Search/Predictive Entropy Search?
What is GPy?
What is GPyOpt?

```

### Tasks

**0. Initialize Gaussian Process**
---

Create the class GaussianProcess that represents a noiseless 1D Gaussian process.

**1. Gaussian Process Prediction**
---

Update the class GaussianProcess:
Public instance method def predict(self, X_s): that predicts the mean and standard deviation of points in a Gaussian process.

**2. Update Gaussian Process**
---

Update the class GaussianProcess.

Public instance method def update(self, X_new, Y_new): that updates a Gaussian Process.

**3. Initialize Bayesian Optimization**
---

Create the class BayesianOptimization that performs Bayesian optimization on a noiseless 1D Gaussian process.

**4. Bayesian Optimization - Acquisition**
---

Update the class BayesianOptimization.

Public instance method def acquisition(self): that calculates the next best sample location.

**5. Bayesian Optimization**
---

Update the class BayesianOptimization.

Public instance method def optimize(self, iterations=100): that optimizes the black-box function.

**6. Bayesian Optimization**
---

Write a python script that optimizes a machine learning model of your choice using GPyOpt.

