#!/usr/bin/env python3
"""
Update training function
train the model using early stopping
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    early_stopping: boolean that indicates whether
                    early stopping should be used
    patience: patience used for early stopping
    """
    callbacks = K.EarlyStopping(patience=patience, monitor='validation_loss')

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,)
    return history
