# 0x04. Convolutions and Pooling
Specializations - Machine Learning ― Math

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### ***General***

___
What is a convolution?
What is max pooling? average pooling?
What is a kernel/filter?
What is padding?
What is “same” padding? “valid” padding?
What is a stride?
What are channels?
How to perform a convolution over an image
How to perform max/average pooling over an image
___


## Tasks

***0. Valid Convolution mandatory***

Write a function def convolve_grayscale_valid(images, kernel): that performs a valid convolution on grayscale images

***1. Same Convolution mandatory***

Write a function def convolve_grayscale_same(images, kernel): that performs a same convolution on grayscale images

***2. Convolution with Padding mandatory***

Write a function def convolve_grayscale_padding(images, kernel, padding): that performs a convolution on grayscale images with custom padding

***3. Strided Convolution mandatory***

Write a function def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on grayscale images

***4. Convolution with Channels mandatory***

Write a function def convolve_channels(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on images with channels

***5. Multiple Kernels mandatory***

Write a function def convolve(images, kernels, padding='same', stride=(1, 1)): that performs a convolution on images using multiple kernels

***6. Pooling mandatory***

Write a function def pool(images, kernel_shape, stride, mode='max'): that performs pooling on images
