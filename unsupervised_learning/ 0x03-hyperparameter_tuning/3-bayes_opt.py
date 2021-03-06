#!/usr/bin/env python3
"""
Class BayesianOptimization
"""
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 L=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        :param self: creat public attributes
        :param f: black-box function to be optimized
        :param X_init: np.ndarray, (t,1) inputs sampled w/
                    black-box function
        :param Y_init: np.ndarray, (t,1) outputs of black-box
                    function for each input
        :param bounds: tuiple, (min, max) bounds of space
                    in which to look for optimal point
        :param ac_samples: num of samples that should be analyzed
                        during acqusition
        :param l: L is used. Cannont compile with var l
                    length parameter for the kernel
        :param sigma_f: standard deviation given output
                    of black-box function
        :param xsi: exploration-exploitation factor for acqusition
        :param minimize: bool, determines to perform optimization for
                    minimization (True) or maximization (False)
        :return:
        """

        self.f = f
        minBounds, maxBounds = bounds
        self.X_s = np.linspace(minBounds, maxBounds,
                               num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.gp = GP(X_init, Y_init, L, sigma_f)
