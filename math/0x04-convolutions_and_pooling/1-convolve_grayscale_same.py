#!/usr/bin/env python3
"""
performs a same convolution on grayscale images:
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images: numpy.ndarray w/ shape (m, h, w)
    kernel: numpy.ndarray w/ shape (kh, kw)
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
    pad_h = int((kh - 1)/2)
    pad_w = int((kw - 1)/2)
    pad_images = np.pad(images, pad_width=((0, 0),
                        (pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)
    cv_output = np.zeros((m, h, w))
    image = np.arange(0, m)
    for y in range(h):
        for x in range(w):
            cv_output[image, y, x] = (np.sum(pad_images
                                             [image, y:kh + y, x:kw + x] *
                                             kernel,
                                             axis=(1, 2)))
    return cv_output
