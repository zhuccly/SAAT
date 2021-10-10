from flags import FLAGS
from ops import *
from utils import *
import tensorflow as tf
slim = tf.contrib.slim

output_size = 256

gf_dim = 64
df_dim = 64
output_c_dim = 3

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn_e2 = batch_norm(name='g_bn_e2')
g_bn_e3 = batch_norm(name='g_bn_e3')
g_bn_e4 = batch_norm(name='g_bn_e4')
g_bn_e5 = batch_norm(name='g_bn_e5')
g_bn_e6 = batch_norm(name='g_bn_e6')
g_bn_e7 = batch_norm(name='g_bn_e7')
g_bn_e8 = batch_norm(name='g_bn_e8')

g_bn_d1 = batch_norm(name='g_bn_d1')
g_bn_d2 = batch_norm(name='g_bn_d2')
g_bn_d3 = batch_norm(name='g_bn_d3')
g_bn_d4 = batch_norm(name='g_bn_d4')
g_bn_d5 = batch_norm(name='g_bn_d5')
g_bn_d6 = batch_norm(name='g_bn_d6')
g_bn_d7 = batch_norm(name='g_bn_d7')




def discriminator(image, y=None, reuse=False):
    batch_size = tf.shape(image)[0]

    with tf.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x df_dim)
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x df_dim*8)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4



def generator(image, y=None):
    batch_size = tf.shape(image)[0]
    with tf.variable_scope("generator") as scope:

        s = output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x gf_dim)
        e2 = g_bn_e2(conv2d(lrelu(e1), gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x gf_dim*2)
        e3 = g_bn_e3(conv2d(lrelu(e2), gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x gf_dim*4)
        e4 = g_bn_e4(conv2d(lrelu(e3), gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x gf_dim*8)
        e5 = g_bn_e5(conv2d(lrelu(e4), gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x gf_dim*8)
        e6 = g_bn_e6(conv2d(lrelu(e5), gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x gf_dim*8)
        e7 = g_bn_e7(conv2d(lrelu(e6), gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x gf_dim*8)
        e8 = g_bn_e8(conv2d(lrelu(e7), gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x gf_dim*8)

        d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8),
                                  [batch_size, s128, s128, gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(g_bn_d1(d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
                                  [batch_size, s64, s64, gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(g_bn_d2(d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
                                  [batch_size, s32, s32, gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(g_bn_d3(d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x gf_dim*8*2)

        d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
                                  [batch_size, s16, s16, gf_dim*8], name='g_d4', with_w=True)
        d4 = g_bn_d4(d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x gf_dim*8*2)

        d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
                                  [batch_size, s8, s8, gf_dim*4], name='g_d5', with_w=True)
        d5 = g_bn_d5(d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x gf_dim*4*2)

        d6, d6_w, d6_b = deconv2d(tf.nn.relu(d5),
                                  [batch_size, s4, s4, gf_dim*2], name='g_d6', with_w=True)
        d6 = g_bn_d6(d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x gf_dim*2*2)

        d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
                                  [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
        d7 = g_bn_d7(d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x gf_dim*1*2)

        d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7),
                                  [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)




def deconv_layer(net, up_scale, n_channel, method='transpose'):
    nh = tf.shape(net)[-3] * up_scale
    nw = tf.shape(net)[-2] * up_scale

    if method == 'transpose':
        net = slim.conv2d_transpose(net, n_channel, (up_scale, up_scale), (
            up_scale, up_scale), activation_fn=None, padding='VALID')
    elif method == 'transpose+conv':
        net = slim.conv2d_transpose(net, n_channel, (up_scale, up_scale), (
            up_scale, up_scale), activation_fn=None, padding='VALID')
        net = slim.conv2d(net, n_channel, (3, 3), (1, 1))
    elif method == 'transpose+conv+relu':
        net = slim.conv2d_transpose(net, n_channel, (up_scale, up_scale), (
            up_scale, up_scale), padding='VALID')
        net = slim.conv2d(net, n_channel, (3, 3), (1, 1))
    elif method == 'bilinear':
        net = tf.image.resize_images(net, [nh, nw])
    else:
        raise Exception('Unrecognised Deconvolution Method: %s' % method)

    return net


def hourglass_arg_scope_tf(weight_decay=0.0001,
                           batch_norm_decay=0.997,
                           batch_norm_epsilon=1e-5,
                           batch_norm_scale=True):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc

def bottleneck_module(inputs, out_channel=256, res=None, scope=''):

    with tf.variable_scope(scope):
        net = slim.stack(inputs, slim.conv2d, [
                         (out_channel // 2, [1, 1]), (out_channel // 2, [3, 3]), (out_channel, [1, 1])], scope='conv')
        if res:
            inputs = slim.conv2d(inputs, res, (1, 1),
                                 scope='bn_res'.format(scope))
        net += inputs

        return net


# recursive hourglass definition
def hourglass_module(inputs, depth=0, deconv='bilinear', bottleneck='bottleneck'):

    bm_fn = globals()['%s_module' % bottleneck]

    with tf.variable_scope('depth_{}'.format(depth)):
        # buttom up layers
        net = slim.max_pool2d(inputs, [2, 2], scope='pool')
        net = slim.stack(net, bm_fn, [
                         (256, None), (256, None), (256, None)], scope='buttom_up')

        # connecting layers
        if depth > 0:
            net = hourglass_module(net, depth=depth - 1, deconv=deconv)
        else:
            net = bm_fn(
                net, out_channel=512, res=512, scope='connecting')

        # top down layers
        net = bm_fn(net, out_channel=512,
                    res=512, scope='top_down')
        net = deconv_layer(net, 2, 512, method=deconv)
        # residual layers
        net += slim.stack(inputs, bm_fn,
                          [(256, None), (256, None), (512, 512)], scope='res')

        return net

# non-stacked hourglass networks
def hourglass(inputs,
              scale=1,
              regression_channels=68,
              deconv='bilinear',
              bottleneck='bottleneck',
              reuse = False
              ):

    with tf.variable_scope('models', reuse=reuse):

        if scale > 1:
            inputs = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
            inputs = slim.layers.avg_pool2d(
                inputs, (3, 3), (scale, scale), padding='VALID')

        output_channels = regression_channels

        with slim.arg_scope(hourglass_arg_scope_tf()):

            # Conv 1
            net = slim.conv2d(inputs, 64, (7, 7), 2, scope='conv1')
            # res
            net = bottleneck_module(net, out_channel=128,
                                    res=128, scope='bottleneck1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # res
            net = slim.stack(net, bottleneck_module, [
                             (128, None), (128, None), (256, 256)], scope='conv2')

            with tf.variable_scope('hourglass'):
                net = hourglass_module(
                    net, depth=4, deconv=deconv, bottleneck=bottleneck)
            net = slim.stack(net, slim.conv2d, [(512, [1, 1]), (256, [1, 1]),(output_channels, [1, 1])
                                                ], scope='conv3')
            net = deconv_layer(net, 2, output_channels, method=deconv)
            net = deconv_layer(net, 2, output_channels, method=deconv)

            with tf.variable_scope('map_lay'):
                net = slim.conv2d(net, output_channels, 1, scope='conv_last')
                regression = slim.conv2d(
                net, regression_channels, 1, activation_fn=None
                ) if regression_channels else None
    return regression



