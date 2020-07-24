#!/usr/bin/env python3
"""
Function calculates the n-gram BLEU score for a sentence
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates n-gram BLEU score for a sentence
    :param references: list of reference translations
    :param sentence: list containing model proposal sentence
    :param n: size of the n-gram to use for evaluation
    :return: n-gram BLEU score
    Note: each rteference translation is a list of words in the translation
    """

    # candidateLen = len(sentence)
    refLen = []
    # candNlen =
    clipped = {}

    # processing the input string
    # for id in range(len(sentence) - n - 1):
    # sentNgrams = [" ".join([str(jd) for j in sentence[id:id + n]])]
    sentNgrams = [' '.join([str(jd) for jd in sentence[id:id + n]])
                  for id in range(len(sentence) - (n - 1))]
    candNlen = (len(sentNgrams))

    # references and sentences
    for refs in references:
        # account for the size of the n-gram
        # for id in range(len(sentence) - n - 1):
        # refNgrams = [" ".join([str(jd) for jdin refs[id:id + n]])]
        refNgrams = [' '.join([str(jd) for jd in refs[id:id + n]])
                     for id in range(len(sentence) - (n - 1))]

        refLen.append(len(refs))

        for w in refNgrams:
            if w in sentNgrams:
                if not clipped.keys() == w:  # clipped dword list
                    clipped[w] = 1
    clipped_count = sum(clipped.values())
    # closest ref length
    closest_refLen = min(refLen, key=lambda m: abs(m - candNlen))
    # brevity penalty
    if candNlen > closest_refLen:
        bp = 1
    else:
        bp = np.exp(1 - (closest_refLen / len(sentence)))
    bleuScore = bp * np.exp(np.log(clipped_count / candNlen))

    return bleuScore
