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
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
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
        self.gp = GP(X_init, self.Y_init, l, sigma_f)

    def acquisition(self):
        """
        calculates next best sample location
        :return: X_next, EI
        """
        m, sigma = self.gp.predict(self.X_s)
        # m_sample, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        # m_opt_sample = np.max(self.X_init)
        if self.minimize is True:
            opt_ptsMin = np.amin(self.gp.Y)
            opt_pts = opt_ptsMin - m - self.xsi
        else:
            opt_ptsMax = np.amax(self.gp.Y)
            opt_pts = m - opt_ptsMax - self.xsi

        with np.errstate(divide='warn'):

            Zed = opt_pts / sigma
            EI = opt_pts * norm.cdf(Zed) + sigma * norm.pdf(Zed)
            # EI[sigma == 0.0] = 0.0

            X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
