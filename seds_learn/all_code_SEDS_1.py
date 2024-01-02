from numpy.random import seed
import tensorflow as tf
# from tensorflow.summary import create_file_writer
# from tensorflow import create_file_writer
import os
import numpy as np
from time import strftime
from nastavenie1 import *

# from lib_hw_SEDS import *

###########################
# Randomness
seed(777)
tf.random.set_seed(222)
###########################

###########################
print("Tensorflow version is: ", tf.__version__)
###########################

##########################
# Data Section
##########################
# Paths to your data files
# Basic dimensions of model
# All in lib_hw_SEDS


##############################
# Model definition
##############################
class HandwritingModel(tf.Module):
    def __init__(self, input_size, hidden0_size, hidden1_size, hidden2_size, output_size):
        super(HandwritingModel, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([input_size, hidden0_size], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([hidden0_size]))

        self.w2 = tf.Variable(tf.random.normal([hidden0_size, hidden1_size], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([hidden1_size]))

        self.w3 = tf.Variable(tf.random.normal([hidden1_size, hidden2_size], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([hidden2_size]))

        self.w4 = tf.Variable(tf.random.normal([hidden2_size, output_size], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([output_size]))

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, TOTAL_INPUTS], dtype=tf.float32)])
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, TOTAL_INPUTS], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool)])
    def __call__(self, x, training=True):
        z1 = tf.matmul(x, self.w1) + self.b1
        a1 = tf.nn.relu(z1)

        # Drop hidden layer
        with tf.name_scope('drop_layer'):
            # dz1 = tf.nn.dropout(a1, rate=0.8, name='dropout_layer')
            dz1 = tf.nn.dropout(a1, rate=0.8, name='dropout_layer') if training else a1

        z2 = tf.matmul(dz1, self.w2) + self.b2
        a2 = tf.nn.relu(z2)

        z3 = tf.matmul(a2, self.w3) + self.b3
        a3 = tf.nn.relu(z3)

        z4 = tf.matmul(a3, self.w4) + self.b4
        return z4

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, NR_CLASSES], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, NR_CLASSES], dtype=tf.float32)])
    def loss_fn(self, logits, labels):
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, NR_CLASSES], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, NR_CLASSES], dtype=tf.float32)])
    def accuracy_fn(self, logits, labels):
        return tf.math.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))


###############################
# Obslužné knižnice
# Re-scale features - aby sme dostali hodnoty 0 - 1 a nie 0 - 255, ako je to teraz
# A One-hot encode správne odpovede, aby sa dali porovnať s výslednými pravdepodobnosťami z predictions
def preprocess_data(x, y):
    # Normalize images
    x = x / 255.0

    # One-hot encode labels
    y = tf.one_hot(y.astype(np.int32), depth=NR_CLASSES)
    return x, y


def get_data():
    # Load data from files
    x_train = np.loadtxt(x_train_path, delimiter=',', dtype=np.float32)
    y_train = np.loadtxt(y_train_path, delimiter=',', dtype=np.int32)
    x_test = np.loadtxt(x_test_path, delimiter=',', dtype=np.float32)
    y_test = np.loadtxt(y_test_path, delimiter=',', dtype=np.int32)
    x_val = np.loadtxt(x_val_path, delimiter=',', dtype=np.float32)
    y_val = np.loadtxt(y_val_path, delimiter=',', dtype=np.int32)
    OBR_VER = np.loadtxt(OBR_VER_PATH, delimiter=',', dtype=np.float32)
    NR_VER = np.loadtxt(NR_VER_PATH, delimiter=',', dtype=np.int32)

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_val, y_val = preprocess_data(x_val, y_val)
    x_test, y_test = preprocess_data(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


# Funkcia vráti batch z dát......
def get_batches(bt_size, x, y):
    for start in range(0, len(x), bt_size):
        end = start + bt_size
        yield x[start:end], y[start:end]


def pisac(model, adres):
    # Get the concrete function
    conc_fn = model.__call__.get_concrete_function()

    # Create a summary file writer
    writer = tf.summary.create_file_writer(adres)

    # Use the writer to write the graph
    with writer.as_default():
        tf.summary.graph(conc_fn.graph)
        writer.flush()


def learning_rate_scheduler(epoch):
    initial_learning_rate = 0.001
    decay_rate = 0.9
    decay_step = 10000
    return initial_learning_rate * decay_rate ** (epoch / decay_step)


def train(model, x_tr, y_tr, x_va, y_va, addr):
    pisac(model, addr)
    # Optimizer
    optimiser = tf.optimizers.Adam()
    # Writer definition
    train_writer = tf.summary.create_file_writer(addr + '/train')
    val_writer = tf.summary.create_file_writer(addr + '/val')

    for epoch in range(nr_epochs):
        total_accuracy = tf.constant(0, dtype=tf.float32)
        num_batches = tf.constant(0, dtype=tf.float32)
        total_loss = tf.constant(0, dtype=tf.float32)
        lr = learning_rate_scheduler(epoch)
        optimiser.learning_rate = lr

        for batch_x, batch_y in get_batches(size_of_batch, x_tr, y_tr):
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = model.loss_fn(logits, batch_y)
                accuracy = model.accuracy_fn(logits, batch_y)

            num_batches += 1
            total_loss += loss
            total_accuracy += accuracy.numpy()

        # Compute gradients
        gradients = tape.gradient(loss,
                                  [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4])
        if any(g is None for g in gradients):
            print(gradients)
            raise ValueError("One or more gradients are None")

        # Apply gradients
        optimiser.apply_gradients(zip(gradients,
                                      [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4]))

        # Average loss
        average_train_loss = total_loss / num_batches
        # Average accuracy
        average_train_accuracy = total_accuracy / num_batches

        with train_writer.as_default():
            for var, varname, grad in zip(
                    [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4],
                    ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4'], gradients):
                tf.summary.histogram(varname, var, step=epoch)
                tf.summary.histogram(f"{varname}/grad", grad, step=epoch)
            # Performance metrics
            with tf.name_scope('performance'):
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy', average_train_accuracy, step=epoch)

        # Validation
        val_logits = model(tf.cast(x_va, tf.float32), training=False)
        val_loss = model.loss_fn(val_logits, y_va)
        val_accuracy = model.accuracy_fn(val_logits, y_va)

        with val_writer.as_default():
            with tf.name_scope('performance'):
                tf.summary.scalar('loss', val_loss, step=epoch)
                tf.summary.scalar('accuracy', val_accuracy, step=epoch)

        print(
            f"Epoch {epoch + 1}, Loss (validation): {val_loss.numpy()}, "
            f"Training Accuracy: {average_train_accuracy * 100:.20f}%, "
            f"Validation Accuracy: {val_accuracy.numpy() * 100:.2f}%")

        print(
            f"Epoch {epoch + 1}, Avg Training Loss: {average_train_loss.numpy()}, Training Accuracy: {average_train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy.numpy() * 100:.2f}%")

        train_writer.flush()
        val_writer.flush()

    #pisac(model, addr)


# Testing
def test_data(model, x_tst, y_tst, batch_sz=100):
    total_correct_predictions = 0
    total_predictions = 0

    for i in range(0, len(x_tst), batch_sz):
        batch_xs = x_tst[i:i + batch_sz]
        batch_ys = y_tst[i:i + batch_sz]

        predictions = model(batch_xs)
        correct_predictions = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(batch_ys, axis=1)), tf.float32))

        total_correct_predictions += correct_predictions.numpy()
        total_predictions += len(batch_xs)

    overall_accuracy = total_correct_predictions / total_predictions
    print(f"Celková presnosť: {overall_accuracy * 100:.2f}%")


def save_model(model, path):
    tf.saved_model.save(model, path)


def load_model(path):
    return tf.saved_model.load(path)


################################
# Začiatok kódu:
# Folder for Tensorboard
folder_name = f'Model 1 at {strftime("%m %d %Y - %H %M")}'
adresar = os.path.join(LOGGING_PATH, folder_name)

try:
    os.makedirs(adresar)
except OSError as exception:
    print(exception.strerror)
else:
    print(f'Successfully created adresar: {adresar}')

x_train, y_train, x_val, y_val, x_test, y_test = get_data()

mod = HandwritingModel(TOTAL_INPUTS, n_hidden0, n_hidden1, n_hidden2, NR_CLASSES)

# Nauč model čisla
train(mod, x_train, y_train, x_val, y_val, adresar)

# Get the concrete function for the model
concrete_function = mod.__call__.get_concrete_function()

# Testing the consistency between direct model call and concrete function
test_data = tf.random.normal([1, TOTAL_INPUTS])
direct_output = mod(test_data, training=False)
concrete_output = concrete_function(test_data, training=False)
print("Direct output:", direct_output)
print("Concrete function output:", concrete_output)

# Save the model
# Folder for Tensorboard
direct = os.path.join(model_path, folder_name)
tf.saved_model.save(mod, direct)
