from numpy.random import seed
seed(888)
import tensorflow as tf
tf.random.set_seed(404)

#from tensorflow.summary import create_file_writer

#import os

#from time import strftime

from seds_learn.nastavenie1 import *


# Model definition
class HandwritingModel(tf.Module):
    def __init__(self, input_size, hidden0_size, hidden1_size, hidden2_size, output_size):
        super(HandwritingModel, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([input_size, hidden1_size], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([hidden1_size]))
        self.w2 = tf.Variable(tf.random.normal([hidden0_size, hidden2_size], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([hidden2_size]))
        self.w3 = tf.Variable(tf.random.normal([hidden1_size, output_size], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([output_size]))
        self.w4 = tf.Variable(tf.random.normal([hidden2_size, output_size], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([output_size]))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, TOTAL_INPUTS], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
        x = tf.nn.relu(tf.matmul(x, self.w3) + self.b3)
        x = tf.matmul(x, self.w4) + self.b4
        return x

    @tf.function
    def loss_fn(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, NR_CLASSES], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, NR_CLASSES], dtype=tf.float32)])
    def accuracy_fn(self, logits, labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))