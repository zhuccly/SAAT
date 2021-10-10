import tensorflow as tf
import models
from flags import FLAGS
import os
slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# general framework
class SAAT(object):

    def __init__(self,  rescale=FLAGS.rescale, batch_size=FLAGS.batch_size, landmark_num=FLAGS.n_landmarks, decay_factor=FLAGS.learning_rate_decay_factor,decay_step = FLAGS.learning_rate_decay_step,learning_rate=FLAGS.initial_learning_rate,is_training=True):

        self.n_channels = landmark_num
        self.batch_size = batch_size
        self.data_path = FLAGS.dataset_dir
        self.rescale = rescale
        self.lr = learning_rate
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        # lambda
        self.lamda = 0.2

    def eval_landmarks(self, inputs,is_training=False):

        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
            with slim.arg_scope(models.hourglass_arg_scope_tf()):
                with tf.device(FLAGS.train_device):
                    prediction = models.hourglass(
                        inputs,
                        regression_channels=self.n_channels,
                        deconv='transpose',
                        bottleneck='bottleneck')

                return prediction

