#!/usr/bin/env python3
"""
Class Bayesian Optimization
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

    def acquisition(self):
        """
        calculates next best sample location
        :return: X_next, EI
        """
        m, sigma = self.gp.predict(self.X_s)
        # sigma = sigma.reshape(-1, 1)
        if self.minimize is True:
            opt_ptsMin = np.min(self.gp.Y)
            opt_pts = opt_ptsMin - m - self.xsi
        else:
            opt_ptsMax = np.max(self.gp.Y)
            opt_pts = m - opt_ptsMax - self.xsi

        with np.errstate(divide='warn'):
            Zed = opt_pts / sigma
            # EI = np.ndarray(ac_samples,)
            EI = (opt_pts * norm.cdf(Zed)) + (sigma * norm.pdf(Zed))
            EI[sigma == 0.0] = 0.0
            X_next = self.X_s[(np.argmax(EI, 0))]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        :param iterations:  max num of iterations to perform
        X_opt: np.ndarray (1,) the optimal point
        Y_opt: np.ndarray (1,) the optimal function
        :return: X_opt, Y_opt
        """

        # sampleY = self.Y_init
        # sampleX = self.X_init
        # ptsUsed = []  # list of used pts

        for m in range(iterations):
            X_next, _ = self.acquisition()
            # compare X_next to list of used pts
            if X_next in self.gp.X:
                break   # if the point has been used stop
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize is True:
            pts = np.argmin(self.gp.Y)
        else:
            pts = np.argmax(self.gp.Y)
        X_opt = self.gp.X[pts]
        Y_opt = self.gp.Y[pts]

        return X_opt, Y_opt
