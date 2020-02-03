#!/usr/bin/env python3
"""Function that calculates the sum of the squares for numbers 1 to n"""


def summation_i_squared(n):
    """sum of squares for numbers 1 to n"""

    if type(n) is not int or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
