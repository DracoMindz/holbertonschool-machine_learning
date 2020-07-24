#!/usr/bin/env python3
"""
Function  calculates cumulative n-gram BLEU score for a sentence
"""
import numpy as np


def cumulative_bleu(references, sentence, n):

    """
    calcualates the n-gram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list contains model prposed sentence
    :param n: size of the largest n-gram to use
                for evaluation
    :return: cumulative n-gram BLEU score
    Note: Each reference translation is a list of words in
        the translation
    Note: All n-gram scores should be weighted evenly
    """




