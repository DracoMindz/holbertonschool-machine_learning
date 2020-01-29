#!/usr/bin/env python3

"""Function adds two matrices"""


def add_matrices2D(mat1, mat2):

    """adds two matrices"""
    if len(mat1[0]) != len(mat2[0]):
        return (None)
    newMat = [[] for i in mat1]
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            newMat[i].append(mat1[i][j] + mat2[i][j])
    return (newMat)
