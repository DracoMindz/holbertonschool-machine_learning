#!/usr/bin/env python3
"""
performs a convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
    ph = padding for height
    pw = padding for width
    sh = stride for height
    sw = stride for width
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[0]
    if padding == 'same':
        ph = int((kh - 1)/2)
        pw = int((kw - 1)/2)
    if padding == 'valid':
        cust_h = h
        cust_w = w
        cpad_images = images
    if type(padding) == 'tuple':
        ph = padding[0]
        pw = padding[1]
    if type(padding) == 'tuple' or padding == 'same':
        cpad_images = np.pad(images, pad_width=((0, 0),
                             (ph, ph), (pw, pw)),
                             mode='constant', constant_values=0)
        cust_h = cpad_images.shape[1]
        cust_w = cpad_images.shape[2]
    cust_sh = int(((cust_h-kh)/sh) + 1)
    cust_sw = int(((cust_w-kw)/sw) + 1)
    cv_output = np.zeros((m, cust_sh, cust_sw))
    image = np.arange(m)
    for y in range(cust_sh):
        for x in range(cust_sw):
            cv_output[image, y, x] = (np.sum(cpad_images
                                      [image, y*sh:(kh + (y*sh)),
                                       x*sw:(kw + (x*sw))] *
                                      kernel,
                                      axis=(1, 2)))
    return cv_output
