#!/usr/bin/env python3
"""
The function calculates the unigram BLEU score for a sentence
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :return: unigram BLEU score
    """

    candidateLen = len(sentence)
    refLen = []
    clipped = {}

    # references and sentences
    for refs in references:
        # refWord_list = np.array(np.abs(len(refs) - candidateLen))
        # refWords_min = np.argwhere(refWord_list == np.min(refWord_list))
        # refWord_len = np.array([len(refs)])[refWords_min]
        # refLen = np.min(refWord_len)  # refers to mo re than one length
        refLen.append(len(refs))

        for w in refs:
            if w in sentence:
                if not clipped.keys() == w:   # clipped dword list
                    clipped[w] = 1
    clipped_count = sum(clipped.values())
    # closest ref length
    closest_refLen = min(refLen, key=lambda m: abs(m - candidateLen))
    # brevity penalty
    if candidateLen > closest_refLen:
        bp = 1
    else:
        bp = np.exp(1 - float(closest_refLen) / float(candidateLen))
    bleuScore = bp * np.exp(np.log(clipped_count / candidateLen))

    return bleuScore
