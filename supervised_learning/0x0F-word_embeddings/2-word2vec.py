#!/usr/bin/env python3
"""
creates and trains a gensim word2vec model
"""
import numpy as np
import tensorflow as tf
import gensim


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0):
    """
    creates and trains a gensim word2vec model
    :param sentences: list of sentences to be trained
    :param size: dim of embedding layer
    :param min_count: min num of occurances of a word
            for use in training
    :param window: max distance betw current and predicted
            word within a sentence
    :param negative: size of negative sampling
    :param cbow: boolean to determine training type
            True: for CBOW
            False: for Skip-gram
    :param iterations: num of iterations to train over
    :param seed: seed for random num generator
    :return: trained model
    """

    if cbow:
        sg = 0
    else:
        sg = 1

    model = gensim.models.Word2Vec(sentences=sentences,
                                   size=size,
                                   min_count=min_count,
                                   window=window,
                                   negative=negative,
                                   sg=sg,
                                   seed=seed,
                                   iter=iterations)

    trained_model = model.wv  # KeyedVectors Instance: trained words stored
    return trained_model
