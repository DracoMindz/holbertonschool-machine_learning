#!/usr/bin/env python3
"""function calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function calculates the integral of a polynomial"""

    coIndex = 0
    dePoly = []

    if len(poly) == 0:
        return None
    # code checking if C is int
    coIndex = 0
    for p, coef in enumerate(poly[1:]):
        if type(coef) is not int or type(coef) is float:
            return None
        if coef != 0:
            coIndex = p + 1
            # change code to C + integral of poly
            dePoly = [C] + 
    

    return dePoly[:coIndex + 1]

def if_coef_whole(num):
    """If a coefficient is a whole number represent as an int"""

    if num.is_integer():
        return int(num)
    else:
        return num
