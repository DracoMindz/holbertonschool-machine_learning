#!/usr/bin/env python3
"""
function conv a gensim word2vec model to keras Embedding layer
"""
import tensorflow.keras as K
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """
    converts gensim word2vec model to keras embedding layer
    :param model: trained gensim word2vec models
    :return: trainable keras embedding
    """
    e_layer = K.layers.Embedding(input_dim,
                                 output_dim=model.size,
                                 embeddings_initializer='uniform',
                                 embeddings_regularizer=None,
                                 activity_regularizer=None,
                                 embeddings_constraint=None,
                                 mask_zero=False, input_length=None)

    e_layerModel = model.wv.get_keras_embedding(train_embeddings=True)
    return e_layerModel
