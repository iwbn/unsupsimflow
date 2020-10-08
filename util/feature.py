import tensorflow as tf
import numpy as np
from util.flow import create_mask

def get_corr_self_flow(corr, max_displacement=4, rate=1):
    d = 2 * max_displacement + 1
    max_arg = tf.math.argmax(corr, axis=-1)[..., tf.newaxis]
    v = tf.div(max_arg, d) - max_displacement
    u = tf.mod(max_arg, d) - max_displacement
    flow = tf.concat([u,v], axis=3) * rate
    flow = tf.cast(flow, tf.float32)
    return flow

def extract_patches(feat1, max_displacement=4, rate=1):
    feat_shape = tf.shape(feat1)
    H = feat_shape[1]
    W = feat_shape[2]
    channel = feat_shape[3]
    d = 2 * max_displacement + 1
    x1_patches = tf.image.extract_patches(feat1, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, rate, rate, 1],
                                          padding='SAME')
    x1_patches = tf.reshape(x1_patches, [-1, H, W, d, d, channel])
    return x1_patches

def correlation(feat1, feat2, max_displacement=4, rate=1):
    feat_shape = tf.shape(feat1)
    H = feat_shape[1]
    W = feat_shape[2]
    channel = feat_shape[3]

    d = 2 * max_displacement + 1
    x2_patches = tf.image.extract_patches(feat2, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, rate, rate, 1],
                                           padding='SAME')
    x2_patches = tf.reshape(x2_patches, [-1, H, W, d, d, channel])
    x1_reshape = tf.reshape(feat1, [-1, H, W, 1, 1, channel])
    x1_dot_x2 = tf.multiply(x1_reshape, x2_patches)
    cost_volume = tf.reduce_sum(x1_dot_x2, axis=-1)
    cost_volume = tf.reshape(cost_volume, [-1, H, W, d * d])
    output = cost_volume
    return output

class Correlation(tf.keras.layers.Layer):
    def __init__(self, max_displacement, rate, **kwargs):
        super(Correlation, self).__init__(**kwargs)
        self.max_displacement = max_displacement
        self.rate = rate

    def build(self, input_shape):
        print(input_shape)
        super(Correlation, self).build(input_shape)

    def __call__(self, inputs, **kwargs):
        a = inputs[0]
        b = inputs[1]
        self.result = correlation(a, b, max_displacement=self.max_displacement, rate=self.rate)
        return self.result

    def compute_output_shape(self, input_shape):
        return tf.keras.int_shape(self.result)


def nhwc_to_nchw(tensor):
    return tf.transpose(tensor, [0, 3, 1, 2])


def nchw_to_nhwc(tensor):
    return tf.transpose(tensor, [0, 2, 3, 1])


def get_corr_flow(feat1, feat2, max_displacement=4, softmax=False, decisive_shape=False):
    with tf.variable_scope("corr_flow"):
        with tf.compat.v1.device("/CPU:0"):
            corr = correlation(feat1, feat2, max_displacement)
        m = tf.cast(tf.meshgrid(np.arange(-max_displacement, max_displacement+1),
                                np.arange(-max_displacement, max_displacement+1)),
                    tf.float32)
        m = tf.transpose(m, [1, 2, 0])
        m = tf.reshape(m, [-1, 2])

        patch_width = max_displacement * 2 + 1
        element_count = patch_width * patch_width

        corr_softmax = tf.nn.softmax(corr, axis=-1)
        corr_r = tf.reshape(corr_softmax, [-1, element_count])

        if decisive_shape:
            hw = corr_softmax.get_shape()[1:3]
        else:
            hw = tf.shape(corr_softmax)[1:3]

        if softmax:
            flow = tf.matmul(corr_r, m)
        else:
            corr_avg = tf.reduce_mean(corr, axis=-1, keepdims=True)
            corr_sub = tf.abs(corr_avg - corr)
            corr_valid = tf.reduce_sum(corr_sub, axis=-1, keepdims=True) > 1e-2

            corr_arg = tf.argmax(corr, axis=-1)
            flow = tf.gather(m, corr_arg, axis=0)
            flow = flow * tf.cast(corr_valid, tf.float32)

        pred_flow_corr = tf.reshape(flow, [-1, hw[0], hw[1], 2])
        output = pred_flow_corr

    tf.add_to_collection("end_points", output)
    return output


def ternary_transform(image, patch_size=7):
    intensities = tf.image.rgb_to_grayscale(image) * 255

    out_channels = patch_size * patch_size
    w = tf.eye(out_channels)
    w = tf.reshape(w, (patch_size, patch_size, 1, out_channels))
    weights = w #tf.constant(w, dtype=tf.float32)
    patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

    transf = patches - intensities
    transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
    return transf_norm


def hamming_distance(t1, t2):
    dist = tf.square(t1 - t2)
    dist_norm = dist / (0.1 + dist)
    dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
    return dist_sum

def corr_entropy(corr):
    corr_sm = tf.nn.softmax(corr, axis=-1)

    corr_ent = -tf.reduce_sum(corr_sm * tf.nn.log_softmax(corr, axis=-1) /
                              tf.math.log(tf.cast(tf.shape(corr)[-1], tf.float32)), axis=-1)

    return corr_ent

def corr_mul(parent, child, max_displacement=4):
    s = tf.shape(child)
    H = s[1]
    W = s[2]
    dim = (2 * max_displacement + 1) ** 2
    parent_r = tf.compat.v2.image.resize(parent, tf.shape(child)[1:3], method='nearest')

    rate = 1
    md = max_displacement
    m = tf.cast(tf.meshgrid(np.arange(-md, md + 1), np.arange(-md, md + 1)), tf.float32)
    m = tf.transpose(m, [1, 2, 0])
    m = tf.reshape(m, [1, 1, 1, -1, 2]) * tf.cast(rate, tf.float32)

    coor_map = tf.cast(tf.meshgrid(tf.range(W), tf.range(H)), tf.int32)
    coor_map = tf.transpose(coor_map, [1, 2, 0])
    coor_map = tf.reshape(coor_map, [1, H, W, 1, 2])

    child_map = tf.cast(tf.floor((tf.cast(coor_map, tf.float32) + tf.cast(m, tf.float32)) / 2.), tf.int32)
    child_map += md
    child_map -= coor_map // 2

    child_map = child_map[..., 1] * (2 * md + 1) + child_map[..., 0]
    child_map = tf.tile(child_map, [s[0],1,1,1])
    parent_g = tf.gather(parent_r, child_map, batch_dims=3, axis=3)
    parent_g = tf.reshape(parent_g, s)

    return parent_g * child


def ternary_transform(image, patch_size=7):
    intensities = tf.image.rgb_to_grayscale(image) * 255
    # patches = tf.extract_image_patches( # fix rows_in is None
    #    intensities,
    #    ksizes=[1, patch_size, patch_size, 1],
    #    strides=[1, 1, 1, 1],
    #    rates=[1, 1, 1, 1],
    #    padding='SAME')
    out_channels = patch_size * patch_size
    w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
    weights = tf.constant(w, dtype=tf.float32)
    patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

    transf = patches - intensities
    transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
    masked = transf_norm * create_mask(transf_norm, [[3, 3], [3, 3]])
    return masked