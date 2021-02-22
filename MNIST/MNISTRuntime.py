import sys

import tensorflow as tf

from MNISTModel import models, fit_and_evaluate
from MNISTData import get_data, expand_data

if not tf.config.list_physical_devices('GPU'):
    print('No GPU is available')
else:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

x_train, y_train, x_test, y_test = get_data()

model = models[sys.argv[1]]()

if sys.argv[1] == 'cnn':
    x_train = expand_data(x_train)
    x_test = expand_data(x_test)

fit_and_evaluate(model, x_train, y_train, x_test, y_test)
