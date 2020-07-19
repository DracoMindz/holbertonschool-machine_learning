#!/usr/bin/env python3
"""
Function creates a bag of words embedding matrix
"""

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
    :param sentences: list  of sentences to analyze
    :param vocab: list of words to use for the analysis
    :return: embeddings, features
    Note: embeddings : np.ndarray shape(s,f)
            s: num of sentences in sentences
            f: num features analyzed
          features: list of rfeatures used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    vector = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = vector.toarray()
    return embeddings, features
