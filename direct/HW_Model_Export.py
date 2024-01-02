from numpy.random import seed
seed(777)
import tensorflow as tf
tf.random.set_seed(505)

from tensorflow.summary import create_file_writer

import os
import numpy as np

from time import strftime
from PIL import Image

#from nastavenie import *
from lib_hd import *


# Folder for Tensorboard
folder_name = f'Model 1 at {strftime("%m %d %Y - %H %M")}'
directory = os.path.join(LOGGING_PATH, folder_name)

try:
    os.makedirs(directory)
except OSError as exception:
    print(exception.strerror)
else:
    print('Succesfully created directory')

x_train, y_train, x_val, y_val, x_test, y_test, obr_ver, nr_ver = get_data()

mod = HandwritingModel(TOTAL_INPUTS, n_hidden1, n_hidden2, NR_CLASSES)

#Train model mod
train(mod, x_train, y_train, x_val, y_val, directory)

# Get the concrete function for the model
concrete_function = mod.__call__.get_concrete_function()

# Testing the consistency between direct model call and concrete function
test_data = tf.random.normal([1, TOTAL_INPUTS])
direct_output = mod(test_data)
concrete_output = concrete_function(test_data)
print("Direct output:", direct_output)
print("Concrete function output:", concrete_output)

# Save the model
tf.saved_model.save(mod, model_path)

# Delete the current model from memory
del mod

# Načítať model
loaded_model = load_model(model_path)
direct_output = loaded_model(test_data)
print("Direct output of loaded model:", direct_output)

"""
x_train, y_train, x_val, y_val, x_test, y_test, obr_ver, nr_ver = get_data()


test_data(loaded_model, x_test, y_test)
test_data(loaded_model, x_val, y_val)
test_data(loaded_model, x_train, y_train)
test_data(loaded_model, obr_ver, nr_ver)

"""