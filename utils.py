from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import menpo.io as mio
from menpo.image import Image
from menpo.shape import PointCloud
import cv2

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables

from menpo.transform import Translation, Scale
from menpo.shape import PointCloud

slim = tf.contrib.slim
jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)

mirrored_parts_68 = np.hstack([
    jaw_indices[::-1], rbrow_indices[::-1], lbrow_indices[::-1],
    upper_nose_indices, lower_nose_indices[::-1],
    np.roll(reye_indices[::-1], 4), np.roll(leye_indices[::-1], 4),
    np.roll(outer_mouth_indices[::-1], 7),
    np.roll(inner_mouth_indices[::-1], 5)
])

def makeGaussian( height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def generate_hm( height, width, joints, maxlenght, weight):
    """ Generate a full Heap Map for every joints in an array
    Args:
        height			: Wanted Height for the Heat Map
        width			: Wanted Width for the Heat Map
        joints			: Array of Joints
        maxlenght		: Lenght of the Bounding Box
    """
    num_joints = joints.shape[0]
    hm = np.zeros((height, width, num_joints), dtype=np.float32)
    for i in range(num_joints):
        if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
            s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
            hm[:, :, i] = makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm

def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color

def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[0:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img

def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])

def keypts_encoding(keypoints, num_classes):
    keypoints = tf.to_int32(keypoints)
    keypoints = tf.reshape(keypoints, (-1,))
    keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=num_classes+1)
    return keypoints

def get_weight(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    weights = tf.where(is_background, ones * ng_w, ones*ps_w)

    return weights

def ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r  = tf.transpose(tf.gather(tf.transpose(dists), [8,12,11,10,2,1,0]))
    pts_l  = tf.transpose(tf.gather(tf.transpose(dists), [9,13,14,15,3,4,5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat(1, [part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[...,None] / tf.shape(dists)[1]])

def pckh(preds, gts, scales):
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2), reduction_indices=-1)) / scales
    return ced_accuracy(0.5, dists)

def import_image(img_path):
    img = cv2.imread(str(img_path))
    original_image = Image.init_from_channels_at_back(img[:,:,-1::-1])

    try:
        original_image_lms = mio.import_landmark_file('{}/{}.ljson'.format(img_path.parent, img_path.stem)).lms.points.astype(np.float32)
        original_image.landmarks['LJSON'] = PointCloud(original_image_lms)
    except:
        pass

    return original_image

def mirror_landmarks_68(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_68])
def mirror_image(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1].copy()

    for group in im.landmarks:

        lms = im.landmarks[group].lms
        if lms.points.shape[0] == 68:
            im.landmarks[group] = mirror_landmarks_68(lms, im.shape)


    return im

def normalized_rmse(pred, gt_truth):
    norm = np.sqrt(np.sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))

    return np.sum(np.sqrt(np.sum(np.square(pred - gt_truth), 2)), 1) / (norm * 68)
def bbx_rmse(pred, gt_truth,x):
    norm = np.sqrt(x)
    return np.sum(np.sqrt(np.sqrt(np.sum(np.square(pred - gt_truth), 2))), 1) / (norm * 68)
def crop_image_bounding_box(img, bbox, res, base=256., order=1):

    center = bbox.centre()
    bmin, bmax = bbox.bounds()
    scale = np.linalg.norm(bmax - bmin) / base

    return crop_image(img, center, scale, res, base, order=order)
def crop_image(img, center, scale, res, base=256., order=1):
    h = scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]).compose_after(Scale((res[0] / h, res[1] / h))).pseudoinverse()

    # Upper left point
    ul = np.floor(t.apply([0, 0]))
    # Bottom right point
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale
    cimg, trans = img.warp_to_shape(
        br - ul, Translation(-(br - ul) / 2 + (br + ul) / 2), return_transform=True)

    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale, order=order).resize(res, order=order)

    trans = trans.compose_after(Scale([c_scale, c_scale]))

    return new_img, trans
def tf_heatmap_to_lms_trick(heatmap):
    hs = tf.argmax(tf.reduce_max(heatmap, 2), 1)
    ws = tf.argmax(tf.reduce_max(heatmap, 1), 1)
    lms = tf.transpose(tf.to_float(tf.stack([hs, ws])), perm=[1, 2, 0])

    return lms
def tf_heatmap_to_lms(heatmap):
    # si = []
    hs = tf.argmax(tf.reduce_max(heatmap, 2), 1)
    hs2 = tf.argmax(tf.reduce_mean(heatmap, 2), 1)
    hs = (hs+hs2)/2
    # print (tf.reduce_max(heatmap, 2).get_shape().as_list())
    # for i in range(68):
    #     a = tf.reduce_max(heatmap, 2)[:,hs[1][i]]+1
    #     b = tf.reduce_max(heatmap, 2)[:,hs[1][i]]-1
    #     c = a-b
    #     d = tf.abs(a-b)
    #     is_background = tf.equal(d, c)
    #     si = si.append(is_background)
    # bb = tf.stack(si,1)
    # ones = tf.to_float(tf.ones_like(bb))
    # weights = tf.where(bb, ones * 1, ones * -1)
    # hs = hs+0.25*weights

    ws = tf.argmax(tf.reduce_max(heatmap, 1), 1)
    ws2 = tf.argmax(tf.reduce_mean(heatmap, 1), 1)
    ws = (ws+ws2)/2

    # a = tf.reduce_max(heatmap, 1)[:, ws + 1,:]
    # b = tf.reduce_max(heatmap, 1)[:, ws - 1,:]
    # c = tf.abs(a - b)
    #
    # is_background = tf.equal(c, c)
    # ones = tf.to_float(tf.ones_like(is_background))
    # weights = tf.where(is_background, ones * 1, ones * -1)
    # ws = ws + 0.25 * weights


    lms = tf.transpose(tf.to_float(tf.stack([hs, ws])), perm=[1, 2, 0])

    return lms