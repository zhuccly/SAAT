import tensorflow as tf
import menpo.io as mio
import numpy as np
import networks
import data_provider
from pathlib import Path
import utils
from menpo.shape import PointCloud
from flags import FLAGS
np.set_printoptions(3)
import datetime
import os
slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.app.flags.DEFINE_string('dataset_path', 'databases/ibug/*.jpg',
                           """The dataset path to evaluate.""")

tf.app.flags.DEFINE_string('bbs_path', 'bbs',
                           """The dataset path to evaluate.""")
tf.app.flags.DEFINE_string('model', 'ckpt/test',
                           """The dataset path to evaluate.""")

def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale

    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im
def evaluate(paths):

    landmarkNum = 68 #68


    binary_mask = np.zeros(landmarkNum)

    mask_index = np.arange(landmarkNum)

    binary_mask[mask_index] = 1

    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    with tf.Graph().as_default() as g:
        images_input = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='input_images')
        net=networks.SAAT(is_training=False)
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=False):
              lms_heatmap_prediction = net.eval_landmarks(images_input,is_training=False)#(images_input, is_training=False)

              pts_predictions = utils.tf_heatmap_to_lms(lms_heatmap_prediction)
              variables_to_restore = slim.get_variables_to_restore()
              saver = tf.train.Saver(variables_to_restore)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config,graph=g)

    saver.restore(sess, ckpt.model_checkpoint_path)

    print ckpt.model_checkpoint_path
    images = data_provider.load_images_test(paths)
    ims = []
    lms = []
    for j in range(len(images)):
        image_test = images[j]

        image_path = image_test.path

        image_test.landmarks['ld'] = mio.import_landmark_file(str(image_path.parent) + '/' + image_path.stem + '.pts')
        bb_root = image_test.path.parent.relative_to(image_test.path.parent.parent.parent)
        if 'set' not in str(bb_root):
            bb_root = image_test.path.parent.relative_to(image_test.path.parent.parent)
        bbox = mio.import_landmark_file(str(Path(
            FLAGS.bbs_path) / bb_root / (image_test.path.stem.replace(' ', '') + '.pts')))
        crop_i, batch_tran = utils.crop_image_bounding_box(image_test, bbox, [256., 256.], base=256. / 256., order=1)
        ims.append(crop_i)
        gt_lms = crop_i.landmarks['ld'].lms.points
        lms.append(gt_lms)
    orig_errors = []
    print datetime.datetime.now()
    for i in range(len(images)):
        crop_i = ims[i]
        gt_lms = lms[i]
        batch_pixels = []
        input_pixels = crop_i.pixels_with_channels_at_back()
        if input_pixels.ndim != 3:
            im = np.expand_dims(input_pixels, 2)
            input_pixels = np.concatenate((im, im, im), 2)
        batch_pixels.append(input_pixels)
        lms_pred = sess.run(
            pts_predictions,
            feed_dict={images_input: np.stack(batch_pixels, axis=0)})
        crop_im = utils.mirror_image(crop_i)
        f_pixels = grey_to_rgb(crop_im).pixels.transpose(1, 2, 0)
        f_pixels = f_pixels.reshape(1, 256, 256, 3)

        flip_pred = sess.run(
            pts_predictions,
            feed_dict={images_input: f_pixels})

        flip_pred = PointCloud(flip_pred)
        flip_pred = utils.mirror_landmarks_68(flip_pred, (256, 256, 3)).points
        flip_pred = np.reshape(flip_pred, (1, 68, 2))

        lms_pred = (lms_pred+flip_pred)/2.
        lms_pred = np.reshape(lms_pred,(1,68,2))

        gt_lms = np.expand_dims(gt_lms, 0)
        crop_rmse = utils.normalized_rmse(lms_pred, gt_lms)
        print (crop_rmse)
        orig_errors.append(crop_rmse)

    orig_errors = np.vstack(orig_errors).ravel()
    print datetime.datetime.now()
    RMSE = orig_errors.mean()
    fr = (orig_errors < .1).mean()

    print fr
    print RMSE

if __name__ == '__main__':
     evaluate(FLAGS.dataset_path.split(':'))


