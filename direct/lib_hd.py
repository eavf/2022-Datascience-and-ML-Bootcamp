from numpy.random import seed

seed(888)
import tensorflow as tf

tf.random.set_seed(404)
from tensorflow import summary
import numpy as np
from model_HandWriting import *


# Re-scale features - aby sme dostali hodnoty 0 - 1 a nie 0 - 255, ako je to teraz
# A One-hot encode správne odpovede, aby sa dali porovnať s výslednými pravdepodobnosťami z predictions 
def preprocess_data(x, y):
    """
    Function to preprocess np arrays of images.
    :param x: tensor dataset with b&w imagearrays
    :param y: tensor dataset with real values
    :return: Normalised datasets to be used with training
    """
    # Normalize images
    x = x / 255.0

    # One-hot encode labels
    y = tf.one_hot(y.astype(np.int32), depth=NR_CLASSES)
    return x, y


def get_data():
    """
    Hunction load MNIST datasets into np arrays. After it preprocess values for training or testing
    :return: set of datasets and corresponding values : train, test, validate and another test
    """
    # Load data from files
    x_train_all = np.loadtxt(x_train_path, delimiter=',', dtype=np.float32)
    y_train_all = np.loadtxt(y_train_path, delimiter=',', dtype=np.int32)
    x_test = np.loadtxt(x_test_path, delimiter=',', dtype=np.float32)
    y_test = np.loadtxt(y_test_path, delimiter=',', dtype=np.int32)
    OBR_VER = np.loadtxt(OBR_VER_PATH, delimiter=',', dtype=np.float32)
    NR_VER = np.loadtxt(NR_VER_PATH, delimiter=',', dtype=np.int32)

    # Validation dataset;
    x_val = x_train_all[:VALIDATION_SIZE]
    y_val = y_train_all[:VALIDATION_SIZE]

    x_train = x_train_all[VALIDATION_SIZE:]
    y_train = y_train_all[VALIDATION_SIZE:]

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_val, y_val = preprocess_data(x_val, y_val)
    x_test, y_test = preprocess_data(x_test, y_test)
    obr_ver, nr_ver = preprocess_data(OBR_VER, NR_VER)
    return x_train, y_train, x_val, y_val, x_test, y_test, obr_ver, nr_ver


# Funkcia vráti batch z dát......
def get_batches(bt_size, x, y):
    """
    Function to get batch of datas from dataset
    :param bt_size: Size of the batch
    :param x: tensor with logits
    :param y: tensor with real values
    :return: batch of data : logits, values
    """
    for start in range(0, len(x), bt_size):
        end = start + bt_size
        yield x[start:end], y[start:end]


def pisac(model, adres):
    """
    Function to write of the Graph of the Tensorflow model for Tensorboard
    :param model: Tensorflow model with 4 hidden layers and one dropout layer
    :param adres: Directory where to write Graph definition
    :return: Nothing
    """
    # Get the concrete function
    concrete_function = model.__call__.get_concrete_function()

    # Create a summary file writer
    writer = summary.create_file_writer(adres)

    # Use the writer to write the graph
    with writer.as_default():
        summary.graph(concrete_function.graph)
        writer.flush()


def train(model, x_tr, y_tr, x_va, y_va, addr):
    """
    Training function designated for specific model in loopin through dataset by defined size of batch of datas.
    :param model: Tensorflow model with 4 hiden layers and 1 dropout layer
    :param x_tr: training dataset with logits
    :param y_tr: training dataset with real values
    :param x_va: validating dataset with logits
    :param y_va: validating dataset with real values
    :param addr: Path, where save the logs from tf.summary.writer
    :return: Nothing
    """
    pisac(model, addr)
    # Optimizer
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    # optimiser = tf.optimizers.Adam()
    # Writer definition
    train_writer = tf.summary.create_file_writer(addr + '/train')
    val_writer = tf.summary.create_file_writer(addr + '/val')

    for epoch in range(nr_epochs):
        total_accuracy = tf.constant(0, dtype=tf.float32)
        num_batches = tf.constant(0, dtype=tf.float32)
        total_loss = tf.constant(0, dtype=tf.float32)
        # lr = learning_rate_scheduler(epoch)
        # optimiser.learning_rate = lr

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


# Testing
def test_data(model, X_tst, Y_tst, batch_sz=100):
    """
    Function testing model accuracy with test dataset
    :param model: tensorflow2 model
    :param X_tst: tensor with dataset with logits
    :param Y_tst: tensor with dataset with real values
    :param batch_sz: size of the batch
    :return: Just print accuracy of the model in percents
    """
    total_correct_predictions = 0
    total_predictions = 0

    for i in range(0, len(X_tst), batch_sz):
        batch_xs = X_tst[i:i + batch_sz]
        batch_ys = Y_tst[i:i + batch_sz]

        predictions = model(batch_xs)
        correct_predictions = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(batch_ys, axis=1)), tf.float32))

        total_correct_predictions += correct_predictions.numpy()
        total_predictions += len(batch_xs)

    overall_accuracy = total_correct_predictions / total_predictions
    print(f"Celková presnosť: {overall_accuracy * 100:.2f}%")


# Testing without batching
def test_data_wo_btch(model, X_tst, Y_tst):
    """
    Function testing model accuracy with test dataset, but this time without batching datasets.
    :param model: tensorflow2 model
    :param X_tst: dataset with logits
    :param Y_tst: dataset with real values
    :return: Accuracy of the model in range from 0 to 1
    """
    predictions = model(X_tst)
    correct_predictions = tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(Y_tst, axis=1)), tf.float32))

    total_correct_predictions = correct_predictions.numpy()
    total_predictions = len(X_tst)

    overall_accuracy = total_correct_predictions / total_predictions
    # print(f"Celková presnosť: {overall_accuracy * 100:.2f}%")
    return overall_accuracy


def save_model(model, path):
    """
    Saving model into designated path
    :param model: Tensorflow2 model
    :param path: directory
    :return: nothing
    """
    tf.saved_model.save(model, path)


def load_model(path):
    """
    Loads model from designated path (directory)
    :param path: directory
    :return: nothing
    """
    return tf.saved_model.load(path)
