#!/usr/bin/env python3
"""
Function creates a TF-IDF embeddings
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates TF-IDF embeddings
    :param sentences: list of sents to analyze
    :param vocab: list of vocabulary words to use for analysis
    :return: embeddings, features
    Note: embeddings: np.ndarray,  shape(s, f) contains embeddings
        s: num of sentences in sentences
        f: num of features analyzed
    Note: features: list of features used for embeddings
    """

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    vector = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = vector.toarray()
    return embeddings, features
