from numpy.random import seed

seed(888)
import tensorflow as tf

tf.random.set_seed(404)
from nastavenie import *

print(tf.__version__)


# Model definition
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
