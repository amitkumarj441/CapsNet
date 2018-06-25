import tensorflow as tf
import os
import time
from tensorflow.python.ops import array_ops

class CapsModel(object):
    def __init__(self, data_provider, params):
        self.data_provider = data_provider
        self.params = params
        self.init_learning_rate = params.init_learning_rate
        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _define_inputs(self):
        self.tensor_input = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1],
                                           name='tensor_input')
        self.tensor_output = tf.placeholder(dtype=tf.float32, shape=[None, 10],
                                            name='tensor_output')

    def _build_graph(self):
        conv1 = self._build_relu_conv1(self.tensor_input)  # shape=[batch_size, 20, 20, 256]
        primary_caps = self._build_primary_caps(conv1)  # shape=[batch_size, 32*6*6, 8]
        digit_caps = self._build_digit_caps(primary_caps)  # shape=[batch_size, 10, 16]
        self.accuracy = self._compute_accuracy(digit_caps)
        tf.summary.scalar('accuracy', self.accuracy)
        self.loss = self._compute_loss(digit_caps)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.init_learning_rate, global_step=global_step,
                                                   decay_rate=0.5, decay_steps=2000, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
        self.summary_all = tf.summary.merge_all()
        print('build model graph done...')

    def _initialize_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.params.log_dir, 'train'), graph=self.sess.graph)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.params.log_dir, 'test'), graph=self.sess.graph)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    def _build_relu_conv1(self, _input):
        """
        Build relu conv1
        :param _input: => [batch_size, 28, 28, 1]
        :return: conv1 => [batch_size, 20, 20, 256]
        """
        with tf.variable_scope('relu_conv1'):
            conv1 = tf.contrib.layers.conv2d(_input, 256, 9, stride=1, padding='VALID')
            conv1 = tf.nn.relu(conv1)
        return conv1

    def _build_primary_caps(self, _input):
        """
        Build primary caps
        :param _input: => [batch_size, 20, 20, 256]
        :return: primary_caps => [batch_size, 32x6x6, 8]
        """
        with tf.variable_scope('primary_caps'):
            conv = tf.contrib.layers.conv2d(_input, 32*8, 9, stride=2, padding='VALID',
                                            activation_fn=None)
            primary_caps = tf.reshape(conv, [-1, 32*6*6, 8])
            primary_caps = self._squashing(primary_caps)
        return primary_caps

    def _build_digit_caps(self, _input):
        """
        Build digit caps
        :param _input: => [batch_size, 32*6*6, 8]
        :return: digit_caps => [batch_size, 10, 16]
        """
        with tf.variable_scope('digit_caps'):
            batch_size = array_ops.shape(_input)[0]
            w = tf.get_variable(name='weights', shape=[1, 32*6*6, 10, 8, 16],
                                initializer=tf.truncated_normal_initializer(stddev=self.params.stddev))
            w = tf.tile(w, multiples=[batch_size, 1, 1, 1, 1])  # shape=[batch_size, 32*6*6, 10, 8, 16]

            input_caps = tf.reshape(_input, [-1, 32*6*6, 1, 1, 8])
            input_caps = tf.tile(input_caps, multiples=[1, 1, 10, 1, 1])  # shape=[batch_size, 32*6*6, 10, 1, 8]
            u = tf.matmul(input_caps, w)  # shape=[batch_size, 32*6*6, 10, 1, 16]
            u = tf.reshape(u, shape=[-1, 32*6*6, 10, 16])
            digit_caps = self._routing(u)
        return digit_caps

    def _compute_loss(self, _input):
        """
        Compute margin loss and reconstruction loss(if used)
        :param _input: digit_prob => [batch_size, 10, 16]
        :return: loss = margin_loss + reconstruction loss
        """
        digit_prob = tf.sqrt(tf.reduce_sum(tf.square(_input), axis=2))  # shape=[batch_size, 10]
        margin_loss1 = tf.reduce_sum(tf.square(tf.maximum(0.0, 0.9-digit_prob)) * self.tensor_output, axis=1)
        margin_loss2 = 0.5 * tf.reduce_sum(tf.square(tf.maximum(0.0, digit_prob-0.1)) * (1.0-self.tensor_output), axis=1)
        self.margin_loss = tf.reduce_mean(margin_loss1 + margin_loss2)
        tf.summary.scalar('margin_loss', self.margin_loss)
        loss = self.margin_loss

        if self.params.reconstruction:
            self.reconstruction_output, self.reconstruction_loss = self._build_reconstruction_graph(_input)
            tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
            loss += self.params.recon_factor * self.reconstruction_loss
        tf.summary.scalar('total_loss', loss)
        return loss

    def _compute_accuracy(self, _input):
        """
        Compute accuracy
        :param _input: digit_caps => [batch_size, 10, 16]
        :return: accuracy => scalar
        """
        digit_prob = tf.sqrt(tf.reduce_sum(tf.square(_input), axis=2))  # shape=[batch_size, 10]
        correct_prediction = tf.equal(tf.argmax(digit_prob, 1), tf.argmax(self.tensor_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _build_reconstruction_graph(self, _input):
        """
        Use an additional reconstruction loss to encourage the digit capsules to encode the instantiation
        parameters of the input digit. During training, we mask out all but the activity vector of the correct
        digit capsule. Then we use this activity vector to reconstruct the input image
        :param _input: digit_caps => [batch_size, 10, 16]
        :return: reconstruction_output => [batch_size, 784]
                  reconstruction_loss => scalar
        """
        def fc_layer(input, units, activation_fn):
            return tf.contrib.layers.fully_connected(input, units, activation_fn=activation_fn)

        with tf.variable_scope('reconstruction_graph'):
            mask = tf.reshape(self.tensor_output, [-1, 10, 1])
            digit_caps_masked = _input * mask
            digit_caps_masked = tf.reshape(digit_caps_masked, [-1, 10*16])
            fc_relu1 = fc_layer(digit_caps_masked, 512, tf.nn.relu)
            fc_relu2 = fc_layer(fc_relu1, 1024, tf.nn.relu)
            reconstruction_output = fc_layer(fc_relu2, 784, tf.nn.sigmoid)

            tensor_input_reshape = tf.reshape(self.tensor_input, [-1, 784])
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction_output-tensor_input_reshape),
                                                               axis=1))
        return reconstruction_output, reconstruction_loss

    def _squashing(self, _input):
        """
        A non-linear "squashing" function to ensure that short vectors get shrunk
        to almost zero length and long vectors get shrunk to a length slightly below 1.
        :param _input: => [batch_size, num_caps, cap_len]
        :return: squashed_caps=> [batch_size, num_caps, cap_len]
        """
        squared_norm = tf.reduce_sum(tf.square(_input), axis=2, keep_dims=True)
        squash_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm)
        squashed_caps = _input * squash_factor
        return squashed_caps

    def _routing(self, _input):
        """
        Routing algorithm
        :param _input: prediction vectors => [batch_size, 32*6*6, 10, 16]
        :return: digit_caps => [batch_size, 10, 16]
        """
        u = _input
        u_temp = tf.stop_gradient(u)
        batch_size = array_ops.shape(_input)[0]
        b = tf.zeros(shape=[batch_size, 32*6*6, 10, 1], dtype=tf.float32)
        for i in range(self.params.iter_routing):
            c = tf.nn.softmax(b)
            if i < self.params.iter_routing-1:
                s = tf.reduce_sum(u_temp*c, axis=1)   # shape=[batch_size, 10, 16]
                v = self._squashing(s)
                # v_temp => [batch_size, 32*6*6, 10, 16]
                v_temp = tf.tile(tf.reshape(v, [-1, 1, 10, 16]), multiples=[1, 32*6*6, 1, 1])
                # u_multi_v => [batch_size, 32*6*6, 10, 1]
                u_multi_v = tf.reduce_sum(u_temp * v_temp, axis=-1, keep_dims=True)
                b = b + u_multi_v
            else:
                s = tf.reduce_sum(u * c, axis=1)  # shape=[batch_size, 10, 16]
                v = self._squashing(s)
        digit_caps = v
        return digit_caps

    def train(self):
        """
        Train capsnet model
        """
        print('training start...')
        batch_size = self.params.batch_size
        train_steps = self.params.train_epoch * self.data_provider.train.num_examples // batch_size
        for step in range(train_steps):
            data, targets = self.data_provider.train.next_batch(batch_size)
            time1 = time.time()
            self.sess.run(self.train_op, feed_dict={self.tensor_input: data, self.tensor_output: targets})
            time_per_batch = time.time() - time1
            if step % 100 == 0:
                fetches = [self.loss, self.accuracy, self.summary_all]
                feed_dict = {self.tensor_input: data, self.tensor_output: targets}
                train_loss, accuracy, merged = self.sess.run(fetches, feed_dict=feed_dict)
                print('train step: {:d}/{:d}, time_per_batch: {:.4f}s, train loss: {:.4f}, accuracy: {:.4f}, '
                      'complete after {:s}'.format(step, train_steps, time_per_batch, train_loss, accuracy,
                                                   get_time_left(time_per_batch, train_steps-step)))
                self.train_writer.add_summary(merged, global_step=step)
            if step % 1000 == 0:
                fetches = [self.loss, self.accuracy, self.summary_all]
                test_data, test_targets = self.data_provider.test.next_batch(1000)
                feed_dict = {self.tensor_input: test_data, self.tensor_output: test_targets}
                test_loss, accuracy, merged = self.sess.run(fetches, feed_dict=feed_dict)
                print('=========== test loss: {:.4f}\taccuracy: {:.4f}============='.format(
                     test_loss, accuracy))
                self.test_writer.add_summary(merged, global_step=step)
                self.saver.save(self.sess, os.path.join(self.params.model_dir, 'model.ckpt'), global_step=step)
        print('training over...')

        test_accuracy = self.sess.run(self.accuracy, feed_dict={self.tensor_input: self.data_provider.test.images,
                                                                self.tensor_output: self.data_provider.test.labels})
        print('final test accuracy: {:.5f}'.format(test_accuracy))

    def test(self):
        """
        Test capsnet model
        """
        batch_size = 100
        total_accuracy = []
        time1 = time.time()
        for i in range(self.data_provider.test.num_examples // batch_size):
            data, targets = self.data_provider.test.next_batch(batch_size)
            accuracy = self.sess.run(self.accuracy,
                                     feed_dict={self.tensor_input: data, self.tensor_output: targets})
            total_accuracy.append(accuracy)
            print('{:d}, acc: {:.4f}'.format(i, accuracy))
        print('test accuracy: {:.5f}, use time: {:.3f}s'.format(sum(total_accuracy)/len(total_accuracy),
                                                                time.time()-time1))

    def load_model(self):
        """
        Load model from trained model files
        """
        last_ckpt = tf.train.latest_checkpoint(self.params.model_dir)
        if not last_ckpt:
            return False
        self.saver.restore(self.sess, last_ckpt)
        print("Successfully load model from save path: {:s}".format(last_ckpt))
        return True


def get_time_left(time_per_batch, batch_num_left):
    seconds = int(time_per_batch * batch_num_left)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:d}:{:d}'.format(h, m, s)
