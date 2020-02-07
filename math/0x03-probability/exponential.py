#!/usr/bin/env python3
"""Class represents an exponential distribution"""


class Exponential:
    def __init__(self, data=None, lambtha=1.):
        """represents exponential distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)
