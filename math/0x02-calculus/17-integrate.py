#!/usr/bin/env python3
"""function calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function calculates the integral of a polynomial"""

    coIndex = 0
    dePoly = []

    try:
        if len(poly) < 1:
            return None
    except TypeError:
        return None
    """code checking if C is int"""
    if not (type(C) is int or type(C) is float):
        return None
    coIndex = 0
    for p, coef in enumerate(poly):
        if not (type(coef) is int or type(coef) is float):
            return None
        if coef != 0:
            coIndex = p + 1
            # print("{}, {}".format(coef, coIndex))
            # change code to C + integral of poly
    # for pow, coef in enumerate(poly):
        # i_pow = coef_whole(coef / (pow + 1))
    # dePoly = ([C] + [i_pow])
    dePoly = [C] + [coef_whole(coef / (exp + 1))
                    for exp, coef in enumerate(poly)]
    #print("{}".format(dePoly))
    return dePoly[:coIndex + 1]


def coef_whole(num):
    """If a coefficient is a whole number represent as an int"""
    if num.is_integer():
        return int(num)
    else:
        return num
