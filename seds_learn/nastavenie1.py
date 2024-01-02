# Paths to your data files
x_train_path = '../datas/SEDS/old/x_train.csv'
y_train_path = '../datas/SEDS/old/y_train.csv'
x_test_path = '../datas/SEDS/old/x_test.csv'
y_test_path = '../datas/SEDS/old/y_test.csv'
x_val_path = '../datas/SEDS/old/x_val.csv'
y_val_path = '../datas/SEDS/old/y_val.csv'
OBR_VER_PATH = '../datas/SEDS/old/x_val.csv'
NR_VER_PATH = '../datas/SEDS/old/y_val.csv'
LOGGING_PATH = 'seds_tfboard_logs'
image_dir = '../SEDS/250000_Final/'
model_path = 'saved_models/'
model_path_seds = 'saved_seds_models/'

NR_CLASSES = 10
VALIDATION_SIZE = 10000
VALIDATION_SIZES = 0.2
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 1
TOTAL_INPUTS = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS


# TensorFlow 2
nr_epochs = 25
learning_rate = 6e-4

# Počet neurónov
n_hidden0 = 2048
n_hidden1 = 512
n_hidden2 = 64

# Set-up učiacej slučky
size_of_batch = 500
batch_size = size_of_batch