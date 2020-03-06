#!/usr/bin/env python3
"""function creates forward propagation graph for NN"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x is placeholder for input data
    layer_sizes is list containing num nodes in each layer of NN
    activations is list containing activation funct 4 each layer
    """

    p_layer = create_layer(x, layer_sizes[0], activations[0])
    for m in range(1, len(layer_sizes)):
        p_layer = create_layer(p_layer, layer_sizes[m], activations[m])
    return p_layer
