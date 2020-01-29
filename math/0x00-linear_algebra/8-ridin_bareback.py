#!/usr/bin/env python3
"""function perfoms matrix multiplication"""


def mat_mul(mat1, mat2):
    """multiply matrices"""
    newMatrix = []
    # length of rows must be equal
    if len(mat1[0]) != len(mat2):
        return None
    for i in range(len(mat1)):
        mRow = []
        for j in range(len(mat2[0])):
            result = 0
            for m in range(len(mat1[0])):
                result += mat1[i][m] * mat2[m][j]
            mRow.append(result)
        newMatrix.append(mRow)
    return(newMatrix)
