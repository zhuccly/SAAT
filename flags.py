import tensorflow as tf

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0002, '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0, '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97, '''Learning rate decay factor.''')
tf.app.flags.DEFINE_float('learning_rate_decay_step', 25000,'''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('batch_size', 8, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,'''How many preprocess threads to use.''')
tf.app.flags.DEFINE_integer('n_landmarks', 68,'''number of landmarks.''')
tf.app.flags.DEFINE_integer('rescale', 256,'''Image scale.''')
tf.app.flags.DEFINE_string('dataset_dir',

            'Train_data/helen.tfrecords,Train_data/lfpw.tfrecords,Train_data/afw.tfrecords',

                           '''Directory where to load datas.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/test', '''Directory where to write event logs and checkpoint.''')
tf.app.flags.DEFINE_string('eval_dir', '','''Directory where to write event logs and checkpoint.''')
tf.app.flags.DEFINE_string('graph_dir', '','''If specified, restore this pretrained model.''')
tf.app.flags.DEFINE_integer('max_steps', 400000,'''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0', '''Device to train with.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', '''Restore pretrained model.''')

