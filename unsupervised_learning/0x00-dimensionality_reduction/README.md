# 0x00. Dimensionality Reduction
Specializations-Machine Learning - Unsupervised Learning

## Learning Objectives
```
What is eigendecomposition?
What is singular value decomposition?
What is the difference between eig and svd?
What is dimensionality reduction and what are its purposes?
What is principal components analysis (PCA)?
What is t-distributed stochastic neighbor embedding (t-SNE)?
What is a manifold?
What is the difference between linear and non-linear dimensionality reduction?
Which techniques are linear/non-linear?
```

### Tasks

***0. PCA***

Write a function def pca(X, var=0.95): that performs PCA on a dataset.

***1. PCA v2***

Write a function def pca(X, ndim): that performs PCA on a dataset.

***2. Initialize t-SNE***

Write a function def P_init(X, perplexity): that initializes all variables required to calculate the P affinities in t-SNE.

***3. Entropy***

Write a function def HP(Di, beta): that calculates the Shannon entropy and P affinities relative to a data point.

***4. P affinities***

Write a function def P_affinities(X, tol=1e-5, perplexity=30.0): that calculates the symmetric P affinities of a data set.

***5. Q affinities***

Write a function def Q_affinities(Y): that calculates the Q affinities.

***6. Gradients***

Write a function def grads(Y, P): that calculates the gradients of Y.

***7. Cost***

Write a function def cost(P, Q): that calculates the cost of the t-SNE transformation.

***8. t-SNE***
Write a function def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500): that performs a t-SNE transformation.


