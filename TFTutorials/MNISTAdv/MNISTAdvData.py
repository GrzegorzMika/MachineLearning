import tensorflow as tf


def get_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(32)
    return train_ds, test_ds
