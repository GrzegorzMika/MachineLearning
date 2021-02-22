import tensorflow as tf

from FashionData import get_data
from FashionModel import build_model, compile_model, fit_and_evaluate

if not tf.config.list_physical_devices('GPU'):
    print('No GPU is available')
else:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

x_train, y_train, x_test, y_test = get_data()

model = build_model()
compile_model(model)
fit_and_evaluate(model, x_train, y_train, x_test, y_test)
