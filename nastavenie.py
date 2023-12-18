# Paths to your data files
X_TRAIN_PATH = 'MNIST//digit_xtrain.csv'
Y_TRAIN_PATH = 'MNIST//digit_ytrain.csv'
X_TEST_PATH = 'MNIST//digit_xtest.csv'
Y_TEST_PATH = 'MNIST//digit_ytest.csv'
LOGGING_PATH = 'tensorboard_minst_digit_logs'


NR_CLASSES = 10
VALIDATION_SIZE = 10000
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNELS = 1
TOTAL_INPUTS = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS

# TensorFlow 2

nr_epochs = 50
learning_rate = 1e-4

# Počet neurónov
n_hidden1 = 512
n_hidden2 = 64