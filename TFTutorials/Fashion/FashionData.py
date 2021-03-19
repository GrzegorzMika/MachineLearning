import tensorflow as tf


def get_data():
    fashion = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test
