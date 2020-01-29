#!/usr/bin/env python3


"""function adds two arrays, element-wise"""


def add_arrays(arr1, arr2):
    """add arrays if same size"""
    if len(arr1) != len(arr2):
        return (None)
    newList = []
    for i in range(len(arr1)):
        newList.append(arr1[i] + arr2[i])
    return (newList)
