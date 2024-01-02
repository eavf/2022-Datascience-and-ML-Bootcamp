# Paths to your data files
x_train_path = '../MNIST/digit_xtrain.csv'
y_train_path = '../MNIST/digit_ytrain.csv'
x_test_path = '../MNIST/digit_xtest.csv'
y_test_path = '../MNIST/digit_ytest.csv'
OBR_VER_PATH = '../MNIST/load_xtest.csv'
NR_VER_PATH = '../MNIST/load_ytest.csv'
LOGGING_PATH = 'tensorboard_minst_digit_logs'
image_directory = 'MNIST/'
model_path = 'saved_models/Model 1 at 01 02 2024 - 21 59'

NR_CLASSES = 10
VALIDATION_SIZE = 10000
VALIDATION_SIZES = 0.2
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNELS = 1
TOTAL_INPUTS = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS

# TensorFlow 2
nr_epochs = 30
learning_rate = 6e-3

# Počet neurónov
n_hidden0 = 1024
n_hidden1 = 512
n_hidden2 = 64

# Set-up učiacej slučky
size_of_batch = 1000
batch_size = size_of_batch