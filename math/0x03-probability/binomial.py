#!/usr/bin/env python3
"""Class represents a binomial distribution"""


class Binomial:
    """Class represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = float(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            vari = sum([(m - mean) ** 2 for m in data]) / len(data)
            self.p = -1 * (vari / mean - 1)
            n = mean / self.p
            self.n = round(n)
            self.p *= n / self.n
