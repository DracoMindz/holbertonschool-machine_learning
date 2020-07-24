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
    nFrac = np.empty((n,))
    nFrac[:] = 1 / n
    bleuScore = bp * np.exp(np.sum(nFrac * np.log(clipped_count / candNlen)))

    return bleuScore
