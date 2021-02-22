import tensorflow as tf
from tensorflow.keras import losses, layers

from TextModel import model, train_ds, val_ds, test_ds, vectorize_layer, raw_test_ds

if not tf.config.list_physical_devices('GPU'):
    print('No GPU is available')
else:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

print('####################################')

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)
