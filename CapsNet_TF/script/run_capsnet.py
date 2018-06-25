from tensorflow.examples.tutorials.mnist import input_data
from capsnet import CapsModel
import tensorflow as tf
import sys

flags = tf.app.flags
flags.DEFINE_string('run_mode', 'train', 'train or test')
flags.DEFINE_string('data_dir', 'MNIST_data', 'mnist data directory')
flags.DEFINE_string('model_dir', 'model_dir', 'model file directory')
flags.DEFINE_string('log_dir', 'log_dir', 'log file directory')
flags.DEFINE_integer('iter_routing', 3, 'iter routing')
flags.DEFINE_bool('reconstruction', True, 'whether use reconstruction regularization')
flags.DEFINE_float('recon_factor', 0.0005, 'reconstruction regularization factor')
flags.DEFINE_float('stddev', 0.01, 'model params init stddev')
flags.DEFINE_float('init_learning_rate', 0.001, 'initial learning rate')
flags.DEFINE_integer('batch_size', 128, 'training batch size')
flags.DEFINE_integer('train_epoch', 30, 'model training epoch')


def main():
    params = flags.FLAGS
    data_provider = input_data.read_data_sets(params.data_dir, one_hot=True,
                                              reshape=False, validation_size=0)
    caps_model = CapsModel(data_provider, params)
    if params.run_mode == 'train':
        caps_model.train()
    if params.run_mode == 'test':
        if not caps_model.load_model():
            print('Training CapsNet model...')
            sys.exit(0)
        caps_model.test()


if __name__ == '__main__':
    main()
