#!/usr/bin/env python3


"""function concatenates two matrices along a specific axis"""

def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along an axis"""
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        newMatrix = []
        newMatrix = mat1 + mat2
        return(newMatrix)
    if axis == 1 and len(mat1) == len(mat2):
        newMatrix = []
        for i in range(len(mat1)):
            m = []
            m.extend(mat1[i])
            m.extend(mat2[i])
            newMatrix.extend([m])
        return(newMatrix)
