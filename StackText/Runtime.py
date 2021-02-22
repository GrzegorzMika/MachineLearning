import tensorflow as tf
import re
import string
from tensorflow.keras import layers, losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing import text_dataset_from_directory

if not tf.config.list_physical_devices('GPU'):
    print('No GPU is available')
else:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

train_ds = text_dataset_from_directory(
    './stack_overflow_16k/train',
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = text_dataset_from_directory(
    './stack_overflow_16k/train',
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=42).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = text_dataset_from_directory(
    './stack_overflow_16k/train',
    batch_size=32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=500
)

# model = tf.keras.Sequential([
#     vectorize_layer,
#     layers.Embedding(10001, 128),
#     layers.Dropout(0.2),
#     layers.GlobalAveragePooling1D(),
#     layers.Dropout(0.2),
#     layers.Dense(4),
#     layers.Activation('softmax')
# ])

model = tf.keras.Sequential([
    vectorize_layer,
    layers.Embedding(10001, 128),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64),
    layers.Dropout(0.2),
    layers.Dense(4),
    layers.Activation('softmax')
])

model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
