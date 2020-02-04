#!/usr/bin/env python3
"""function calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function calculates the integral of a polynomial"""

    coIndex = 0
    dePoly = []

    if len(poly) == 0 or len(poly) < 1:
        return None
    """code checking if C is int"""
    if type(C) is not int or type(C) is float:
        return None
    coIndex = 0
    for p, coef in enumerate(poly):
        if type(coef) is not int or type(coef) is float:
            return None
        if coef != 0:
            coIndex = p + 1
            # change code to C + integral of poly
        for pow, coef in enumerated(poly):
            i_pow = (pow + 1)
    dePoly = ([C] + (coef_whole(coef / i_pow))
    return dePoly[:coIndex + 1]



def coef_whole(num):
    """If a coefficient is a whole number represent as an int"""
    if num.is_integer():
        return int(num)
    else:
        return num
