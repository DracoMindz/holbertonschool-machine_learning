#!/usr/bin/env python3
"""
Function creates and trains a gensim fastText model
"""

import numpy as np
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    creataes and trains a gensim fastText model
    :param sentences: list of sents to be trained on
    :param size: dim of the embedding layer
    :param min_count: min_count min num of occurences of a word
    for use in training
    :param negative: size of negative sampling
    :param window: max distance betw current and predicted word
    within a sentence
    :param cbow: boolean to determine training type
            True: CBOW
            False: Skip-gram
    :param iterations: num of iterations to train ove
    :param seed: seed for the random num generator
    :param workers: num of worker threads to train model
    :return: trained model
    """
    if cbow:
        sg = 0
    else:
        sg = 1

    model = FastText(sentences=sentences,
                     size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=sg,
                     seed=seed,
                     iter=iterations)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
