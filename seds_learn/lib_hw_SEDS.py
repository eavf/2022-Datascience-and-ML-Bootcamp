from numpy.random import seed
seed(888)
import tensorflow as tf
tf.random.set_seed(404)

from tensorflow.summary import create_file_writer

from model_hw_seds import *


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
def get_batches(batch_size, x, y):
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        yield x[start:end], y[start:end]


def pisac(model, dir):
    # Get the concrete function
    concrete_function = model.__call__.get_concrete_function()

    # Create a summary file writer
    writer = tf.summary.create_file_writer(dir)

    # Use the writer to write the graph
    with writer.as_default():
        tf.summary.graph(concrete_function.graph)
        writer.flush()


def train (model, x_tr, y_tr, x_va, y_va, dir):
    pisac(model, dir)
    # Optimizer
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    # Writer definition
    train_writer = create_file_writer(dir + '/train')
    val_writer = create_file_writer(dir + '/val')

    for epoch in range(nr_epochs):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch_x, batch_y in get_batches(size_of_batch, x_tr, y_tr):
            with tf.GradientTape() as tape:
                logits = model(batch_x)
                loss = model.loss_fn(logits, batch_y)
                accuracy = model.accuracy_fn(logits, batch_y)

            total_loss += loss.numpy()
            total_accuracy += accuracy.numpy()
            num_batches += 1

        # Compute gradients
        gradients = tape.gradient(loss, [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3])
        if any(g is None for g in gradients):
            print(gradients)
            raise ValueError("One or more gradients are None")

        # Apply gradients
        optimiser.apply_gradients(zip(gradients, [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]))

        # Average loss
        average_train_loss = total_loss / num_batches
        # Average accuracy
        average_train_accuracy = total_accuracy / num_batches
            
        with train_writer.as_default():
            for var, varname, grad in zip([model.w1, model.b1, model.w2, model.b2, model.w3, model.b3], ['w1', 'b1', 'w2', 'b2', 'w3', 'b3'], gradients):
                tf.summary.histogram(varname, var, step=epoch)
                tf.summary.histogram(f"{varname}/grad", grad, step=epoch)
            # Performance metrics
            with tf.name_scope('performance'):
                tf.summary.scalar('loss', average_train_loss, step=epoch)
                tf.summary.scalar('accuracy', average_train_accuracy, step=epoch)

        num_batches += 1

        # Validation
        val_logits = model(tf.cast(x_va, tf.float32))
        val_loss = model.loss_fn(val_logits, y_va)
        val_accuracy = model.accuracy_fn(val_logits, y_va)
        
        with val_writer.as_default():
            with tf.name_scope('performance'):
                tf.summary.scalar('accuracy', val_accuracy, step=epoch)
                tf.summary.scalar('cost', val_loss, step=epoch)
        
        print(f"Epoch {epoch + 1}, Loss (validation): {val_loss.numpy()}, Training Accuracy: {average_train_accuracy * 100:.20f}%, Validation Accuracy: {val_accuracy.numpy() * 100:.2f}%")

        train_writer.flush()
        val_writer.flush()

    pisac(model, dir)





# Testing
def test_data(model, x_test, y_test, batch_size=100):
    total_correct_predictions = 0
    total_predictions = 0

    for i in range(0, len(x_test), batch_size):
        batch_xs = x_test[i:i + batch_size]
        batch_ys = y_test[i:i + batch_size]

        predictions = model(batch_xs)
        correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(batch_ys, axis=1)), tf.float32))

        total_correct_predictions += correct_predictions.numpy()
        total_predictions += len(batch_xs)

    overall_accuracy = total_correct_predictions / total_predictions
    print(f"Celková presnosť: {overall_accuracy * 100:.2f}%")


def save_model(model, path):
    tf.saved_model.save(model, path)


def load_model(path):
    return tf.saved_model.load(path)