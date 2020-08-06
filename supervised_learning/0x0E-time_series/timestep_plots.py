#!/usr/bin/env python3
"""
functions for plotting and creating time steps
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def create_timeSteps(length):
    """ create timestep"""
    return list(range(-length, 0))


def plot_trainHistory(history, title):
    """ plot data history"""
    loss = history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(title)
    plt.legend()

    plt.show()


def baseline(history):
    """ history mean"""
    return np.mean(history)


def show_plot(plot_data, delta, title):
    """ show plots of points"""
    labels = ['History', 'Real Future', 'model Prediction']
    marker = ['.-', 'rx', 'go']
    timeSteps = create_timeSteps(plot_data[0].shape[0])
    if delta:
        rFuture = delta
    else:
        rFuture = 0
    plt.title(title)

    for idx, x in enumerate(plot_data):
        if idx:
            plt.plot(rFuture, plot_data[idx],
                     marker[idx], markersize=10,
                     label=labels[idx])
        else:
            plt.plot(timeSteps, plot_data[idx].flatten(),
                     marker[idx], label=labels[idx])
    plt.legend()
    plt.xlim([timeSteps[0], (rFuture+5)*2])
    plt.xlabel('Time-Step')

    return plt
