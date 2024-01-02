from numpy.random import seed
seed(888)
import tensorflow as tf
tf.random.set_seed(404)

from tensorflow.summary import create_file_writer

import os
import numpy as np

from time import strftime
from PIL import Image

from nastavenie1 import *

loaded_model = tf.saved_model.load(model_path)

# List all available signatures (callable functions)
for key in loaded_model.signatures.keys():
    print(f"Signature: {key}")
    concrete_func = loaded_model.signatures[key]

    # Print variables for each concrete function
    for var in concrete_func.graph.get_operations():
        print(var.name, var.type)

