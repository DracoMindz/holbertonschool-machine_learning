#!/usr/bin/env python3


def matrix_transpose(matrix):
    """transpose matrix"""
    m = []
    for i in range(len(matrix[0])):
        mx = []
        for j in range(len(matrix)):
            mx.append(matrix[j][i])
        m.append(mx)
    return(m)
