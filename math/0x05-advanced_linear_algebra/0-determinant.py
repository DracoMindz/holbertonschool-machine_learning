#!/usr/bin/env python3
"""
function that calculates the determinant of a matrix
"""


def determinant(matrix):
    """
    Function calculates the determinant of a matrix
    :param matrix: list of lists, determinant to be calculated
    :return: determinant of matrix
    """
    tot_returns = 0

    # check if is list and check if is square matrix
    if not isinstance(matrix, list) or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    if ((len(matrix) == 1 and isinstance(matrix[0], list))
            and len(matrix[0]) == 0):
        return 1
    for m in matrix:
        if not isinstance(m, list):
            raise TypeError("matrix must be a list of lists")
        if len(m) != len(matrix[0]) or len(m) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    # if len(matrix) != len(matrix[0]):
        # raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]

    # working with 2X2 submatrices, then end
    if len(matrix) == 2 and len(matrix[0]) == 2:
        matrix_two = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return matrix_two

    # define submatric for each column
    for fCol in list(range(len(matrix))):
        matCopy = matrix_copy(matrix)
        matCopy = matCopy[1:]  # remove first row
        mheight = len(matCopy)

        # remaining submatrix
        for idx in range(mheight):
            matCopy[idx] = matCopy[idx][0:fCol] + matCopy[idx][fCol+1:]

        # signs for submatrix multiplier
        sign = (-1) ** (fCol % 2)
        subMatrix_Det = determinant(matCopy)  # pass recursively
        tot_returns += sign * matrix[0][fCol] * subMatrix_Det
    return tot_returns


def zeroMatrix(rows, cols):
    """
    New matrix filled with zeros
    contains row and columns
    :return: matrix
    """
    matrixZ = []
    while len(matrixZ) < rows:
        matrixZ.append([])
        while len(matrixZ[-1]) < cols:
            matrixZ[-1].append(0.0)
    return matrixZ


def matrix_copy(matrixA):
    """
    copy matrix
    :param matrixA: matrix to be copied
    :return: A copy of a matrix
    """
    # matrix dimensions
    rows = len(matrixA)
    cols = len(matrixA[0])

    # new zero filled matrix
    matrixCopy = zeroMatrix(rows, cols)

    # copy value into the zero matrix
    for i in range(rows):
        for j in range(cols):
            matrixCopy[i][j] = matrixA[i][j]
    return matrixCopy
