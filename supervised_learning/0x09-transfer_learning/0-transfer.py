#!/usr/bin/env python3
"""
Write a python script that trains a convolutional
neural network to classify the CIFAR 10 dataset
"""

import tensorflow.keras as K


def load_data():
    """ load data: cifar10 """
    # import data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Normalize values to range between 0 & 1
    # Change integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot target values
    Y_train = K.utils.to_categorical(Y_train, 10)
    Y_test = K.utils.to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test


def model_def(Y_train):
    """ model based on VGG16 architecture for CIFAR10 """
    # use VGG16 for bottlensck features
    vgg_bf = K.applications.vgg16.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_tensor=K.Input(
                                            shape=(32, 32, 3)),
                                        classes=Y_train.shape[1])
    for layer in vgg_bf.layers:
        # get layers before fully connected layers
        if (layer.name[0:5] != 'block'):
            layer.trainiable = False

    modelY = K.Sequential()

    modelY.add(vgg_bf)
    modelY.add(K.layers.Flatten())
    modelY.add(K.layers.Dense(256, activation='relu',
                              kernel_initializer='he_uniform'))
    modelY.add(K.layers.Dense(10, activation="softmax"))
    modelY.summary()
    return modelY


def compile_model(new_cnn):
    """ compile mode """
    """ compile model separate, vgg_bf classes=Y_train.shape[1] """

    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    new_cnn.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return new_cnn


def train_model(new_cnn, X_train, Y_train, X_test, Y_test, batch, epochs):

    dataGen = K.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                       width_shift_range=0.1,
                                                       height_shift_range=0.1,
                                                       horizontal_flip=True)
    dataGen.fit(X_train)
    return new_cnn.fit_generator(dataGen.flow(X_train, Y_train,
                                              batch_size=batch),
                                 steps_per_epoch=X_train.shape[0] / batch,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(X_test, Y_test))


def preprocess_data(X, Y):
    """
    X: numpy.ndarray, shape(m, 32, 32, 3) containing CIFAR 10 data
    m: number of data points
    Y: numpy.ndarray, shape(m, ) containing the CIFAR 10 labels for X
    X_p: numpy.ndarray containing preprocessed X
    Y_p: numpy.ndarray containing preprocessed Y
    Return: X_p, Y_p

    trained model:  save in directory as cifar10.h5
                    should be compiled
                    validation accuracy of 88% or higher
    file script should not run when file is imported
    """

    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


if __name__ == '__main__':
    # step by step make

    batch = 50
    epochs = 50

    X_train, Y_train, X_test, Y_test = load_data()
    t_model = model_def(Y_train)
    t_model = compile_model(t_model)
    history = train_model(t_model, X_train, Y_train,
                          X_test, Y_test, batch, epochs)
    t_model.save('cifar10.h5')
