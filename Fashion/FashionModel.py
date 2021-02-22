import datetime

import tensorflow as tf


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


def fit_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=100):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3, verbose=0,
        mode='auto', baseline=None, restore_best_weights=False
    )

    checkpoint_filepath = 'logs/checkpoint/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback, early_stopping, model_checkpoint_callback])

    loss, acc = model.evaluate(x_test, y_test)
    print(f"Model test accuracy: {acc}")
