#!/usr/bin/env python3
"""function perfoms matrix multiplication"""


def mat_mul(mat1, mat2):
    """multiply matrices"""
    if len(mat1[0]) != len(mat2):
        return None

    newMatrix = [[] for i in mat1]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            result = 0
            for m in range(len(mat1[0])):
                result += mat1[i][m] * mat2[m][i]
                newMatrix[i].append(result)
    return(newMatrix)
