# 0x05. Generative Adversarial Networks

 Specializations - Machine Learning ― Unsupervised Learning 
 
 ##Learning Objectives:

    What is a generator?
    What is a discriminator?
    What is the minimax loss? modified minimax loss? wasserstein loss?
    How do you train a GAN?
    What are the use cases for GANs?
    What are the shortcoming of GANs?

###Tasks

**0. Generator**
--
Write a function def generator(Z): 
that creates a simple generator network for MNIST digits.

**1. Discriminator**
--
Write a function def discriminator(X): 
that creates a discriminator network for MNIST digits.

**2. Train Discriminator**
--
Write a function def train_discriminator(Z, X):
that creates the loss tensor and training op for the discriminator.

**3. Train Generator mandatory**
--
Write a function def train_generator(Z):
that creates the loss tensor and training op for the generator.

**4. Sample Z mandatory**
--
Write a function def sample_Z(m, n):
that creates input for the generator.

**5. Train GAN mandatory**
--
Write a function def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):
that trains a GAN.
