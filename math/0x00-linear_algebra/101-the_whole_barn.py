#!/usr/bin/env python3
"""function perfoms matrix addition"""


def add_matrices(mat1, mat2):

    """adds two matrices"""
    try:
        if len(mat1) != len(mat2):
            return None
        newMat = []
        for i, j in zip(mat1, mat2):
            newAdd = add_matrices(i, j)
            if newAdd is None:
                return None
            newMat.append(newAdd)
        return (newMat)
    except TypeError:
        return mat1 + mat2
