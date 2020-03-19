#!/usr/bin/env python3
"""
performs a convolution on grayscale images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images: numpy.ndarray w/ shape (m, h, w)
    kernel: numpy.ndarray w/ shape (kh, kw)
    padding: tuple of (ph, pw)
    Returns: a numpy.ndarray cotaining convolved images
    m: num of images
    h: height in pixels of images
    w: width in pixels of images
    kh: height of the kernel
    kw: width of the kernel
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = int((kh - 1)/2)
    pw = int((kw - 1)/2)
    cpad_images = np.pad(images, pad_width=((0, 0),
                         (ph, ph), (pw, pw)),
                         mode='constant', constant_values=0)
    cust_h = cpad_images.shape[1]
    cust_w = cpad_images.shape[2]
    cv_output = np.zeros((m, cust_h-kh+1, cust_w-kh+1))
    image = np.arange(m)
    for y in range(cust_h-kh+1):
        for x in range(cust_w-kh+1):
            cv_output[image, y, x] = (np.sum(cpad_images
                                      [image, y:kh + y, x:kw + x] *
                                      kernel,
                                      axis=(1, 2)))
    return cv_output
