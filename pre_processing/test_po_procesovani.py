import pandas as pd
import numpy as np
from PIL import Image


# Paths to your data files
x_train_path = '../datas/SEDS/x_train.csv'
y_train_path = '../datas/SEDS/y_train.csv'
x_test_path = '../datas/SEDS/x_test.csv'
y_test_path = '../datas/SEDS/y_test.csv'
x_val_path = '../datas/SEDS/x_val.csv'
y_val_path = '../datas/SEDS/y_val.csv'
OBR_VER_PATH = '../datas/SEDS/x_val.csv'
NR_VER_PATH = '../datas/SEDS/y_val.csv'
LOGGING_PATH = 'seds_tfboard_logs'
image_dir = '../SEDS/250000_Final/'
model_path = 'saved_models/'
model_path_seds = 'saved_seds_models/'


# Function to save images from array
def save_images_from_array(image_arrays, labels, output_folder, img_size):
    for idx, img_array in enumerate(image_arrays):
        # Reshape the array into an image format (e.g., 28x28 for MNIST)
        image = img_array.reshape(img_size, img_size)
        # Convert to uint8 format
        image = np.uint8(image)
        # Create an image object
        img = Image.fromarray(image, 'L')  # 'L' mode for grayscale
        # Save the image with a filename based on its label
        img.save(f"{output_folder}/{idx}_{labels[idx]}.png")


def get_data():
    # Load data from files
    x_train = np.loadtxt(x_train_path, delimiter=',', dtype=np.float32)
    y_train = np.loadtxt(y_train_path, delimiter=',', dtype=np.int32)
    x_test = np.loadtxt(x_test_path, delimiter=',', dtype=np.float32)
    y_test = np.loadtxt(y_test_path, delimiter=',', dtype=np.int32)
    x_val = np.loadtxt(x_val_path, delimiter=',', dtype=np.float32)
    y_val = np.loadtxt(y_val_path, delimiter=',', dtype=np.int32)
    return x_val, y_val
    # return x_train, y_train, x_val, y_val, x_test, y_test


# Load data from CSV
x_data, y_labels = get_data()

# Define the size of the images (e.g., 28 for 28x28 images)
img_size = 64

# Output folder where images will be saved
output_folder = 'imgs'

# Save images from array
save_images_from_array(x_data, y_labels, output_folder, img_size)
