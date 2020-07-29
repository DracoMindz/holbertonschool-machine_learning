#!/usr/bin/env python3
"""
Function that calcuylates the scaled dot product attention
"""

import numpy as np
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates scaled dot product attention
    :param Q: tensor w/ last two dims as (..., seq_len_q, dk)
            contains query matrix
    :param K: tensor w? last two dims as (..., seq_len_v, dk)
            contains key matrix
    :param V: tensor w/ last two dims as (.., seq_len_v, dv)
            contains value matrix
    :param mask: tensor can broadcast into (..., seq_len_q, seq_len_v)
            contains optional mask or defaulted to NONE
    Note: if mask is none, nult -1e9 to mask and add it to ths scaled matrix mult
    Note: Q, K, V dimemsions are the same
    :return: outputs, weights
        outputs: tensor w/ last two dims as (..., seq_len_q, dv)
                contains: scaled dot prod attention
        weights: tensor w/ last vtwo dims as (..., seq_len__q, seq_len_v)
                contains: attention weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    scald_qk =tf.cast(tf.shape(K)[-1], tf.float32)
    scald_atten = matmul_qk / tf.math.sqrt(dk)

    # masked added to tensor
    if mask is not None:
        scald_attention += (mask * -1e9)
    # weights are the attention weights
    weights = tf.nn.softmax(scaled_atten, axis=-1)
    output = tf.matmul(weights, V)

    return outputs, weights
