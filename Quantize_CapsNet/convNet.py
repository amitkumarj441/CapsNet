import numpy as np
import tensorflow as tf

from config import cfg
from utils import get_batch_data, quantize


class ConvNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))
                self.build_arch()

        tf.logging.info('Setting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 28, 28, 32]
            W1 = tf.get_variable('Weight1', shape=(5, 5, 1, 32), dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=cfg.stddev))
            biases1 = tf.get_variable('bias1', shape=(32))

            '''
            if not cfg.is_training:
                W1 = quantize(W1, cfg.bits)
                biases1 = quantize(biases1, cfg.bits)
            '''

            conv1 = tf.nn.relu(tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME') + biases1)
            assert conv1.get_shape() == [cfg.batch_size, 28, 28, 32]

        with tf.variable_scope('Pooling1_layer'):
            # Pooling1, [batch_size, 14, 14, 32]
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            assert pool1.get_shape() == [cfg.batch_size, 14, 14, 32]

        with tf.variable_scope('Conv2_layer'):
            # Conv2, [batch_size, 14, 14, 64]
            W2 = tf.get_variable('Weight2', shape=(5, 5, 32, 64), dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=cfg.stddev))
            biases2 = tf.get_variable('bias1', shape=(64))

            '''
            if not cfg.is_training:
                W2 = quantize(W2, cfg.bits)
                biases2 = quantize(biases2, cfg.bits)
            '''

            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME') + biases2)
            assert conv2.get_shape() == [cfg.batch_size, 14, 14, 64]

        with tf.variable_scope('Pooling2_layer'):
            # Pooling1, [batch_size, 7, 7, 64]
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            assert pool2.get_shape() == [cfg.batch_size, 7, 7, 64]

        with tf.variable_scope('FC1_layer'):
            # FC1, [batch_size, 1024]
            W3 = tf.get_variable('Weight3', shape=(7 * 7 * 64, 1024), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
            biases3 = tf.get_variable('bias3', shape=(1024))

            if not cfg.is_training:
                W3 = quantize(W3, cfg.bits)
                biases3 = quantize(biases3, cfg.bits)

            flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])
            fc1 = tf.nn.relu(tf.matmul(flatten, W3) + biases3)
            assert fc1.get_shape() == [cfg.batch_size, 1024]

        with tf.variable_scope('Dropout'):
            keep_prob = 0.5 if cfg.is_training else 1.0
            dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)

        with tf.variable_scope('FC2_layer'):
            # FC1, [batch_size, 1024]
            W4 = tf.get_variable('Weight4', shape=(1024, 10), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
            biases4 = tf.get_variable('bias4', shape=(10))

            if not cfg.is_training:
                W4 = quantize(W4, cfg.bits)
                biases4 = quantize(biases4, cfg.bits)

            # self.softmax_v = tf.nn.softmax(tf.matmul(dropout, W4) + biases4)
            self.softmax_v = tf.matmul(dropout, W4) + biases4
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10]

        self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
        assert self.argmax_idx.get_shape() == [cfg.batch_size]

    def loss(self):
        self.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.softmax_v))

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
