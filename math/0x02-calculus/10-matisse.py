#!/usr/bin/env python3
"""function calculates derivative of a polynomial"""


def poly_derivative(poly):
    """function calculates derivative of a polnomial"""

    coIndex = 0
    dePoly = []

    if len(poly) == 0:
        return None

    for p, c in enumerate(poly[1:]):
        if type(c) is not int or type(c) is float:
            return None
        if c != 0:
            coIndex = p
        dePoly.append(c * (p + 1))
    if coIndex == 0:
        return [0]
    return dePoly[:coIndex + 1]
