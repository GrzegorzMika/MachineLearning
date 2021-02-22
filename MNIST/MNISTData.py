from tensorflow import keras as k
import numpy as np


def preprocess_data(X):
    return X / 255.0


def expand_data(X):
    return np.expand_dims(X, -1)


def get_data():
    (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    return x_train, y_train, x_test, y_test
