import tensorflow as tf
import numpy as np
import math
import util.image
import tensorflow_addons as tfa

def visualize_flow(flow, max_mag=None):
    s = tf.shape(flow)
    # Use Hue, Saturation, Value colour model
    hsv_2 = tf.ones([s[0], s[1]], dtype=flow.dtype)

    x = flow[:, :, 0]
    y = flow[:, :, 1]
    rho = tf.sqrt(x ** 2 + y ** 2)
    phi = tf.numpy_function(lambda x, y: np.arctan2(y, x, dtype=y.dtype), [x, y], flow.dtype)
    phi = tf.where(tf.less(phi, 0), phi + 2. * math.pi, phi)
    if max_mag:
        rho = rho / max_mag
        rho = tf.clip_by_value(rho, 0., 1.)
    else:
        max_mag = tf.reduce_max(rho)
        max_mag = tf.cond(tf.equal(max_mag, 0.), lambda: tf.constant(1., dtype=max_mag.dtype), lambda: max_mag)
        rho = rho / max_mag

    hsv_0 = tf.cast(phi / (2. * math.pi), flow.dtype)
    hsv_1 = tf.cast(rho, flow.dtype)
    hsv = tf.stack([hsv_0, hsv_1, hsv_2], axis=2)
    rgb = tf.image.hsv_to_rgb(hsv)
    return rgb

def visualize_flows(flows, max_mag=None):
    rgbs = tf.map_fn(lambda x: visualize_flow(x, max_mag), flows, dtype=flows.dtype)
    return rgbs


def resize_flow(flow, size, scaling=True, align_corners=False):
    flow_size = tf.cast(tf.shape(flow)[-3:-1], tf.float32)

    assert (align_corners == False)
    flow = util.image.resize_bilinear(flow, size)
    if scaling:
        scale = tf.cast(size, tf.float32) / flow_size
        scale = tf.stack([scale[1], scale[0]])
        scale = tf.reshape(scale, [1,1,1,2])
        flow = tf.multiply(flow, scale)
    return flow


def crop_both(im, flow, start_pos, size):
    im = im[:, start_pos[0]:start_pos[0]+size[0], start_pos[1]:start_pos[1]+size[1],:]
    flow = flow[:, start_pos[0]:start_pos[0]+size[0], start_pos[1]:start_pos[1]+size[1],:]
    return im, flow

def rand_crop_both(im, flow, size):
    ims = tf.shape(im)

    rand_mh, rand_mw = ims[1] - size[0], ims[2] - size[1]
    _rand_func = lambda max_val: tf.random.uniform([], 0, max_val+1, dtype=tf.int32)
    rand_h, rand_w = _rand_func(rand_mh), _rand_func(rand_mw)
    return crop_both(im, flow, (rand_h, rand_w), size)

def random_channel_swap(im):
    channel_permutation = tf.constant([[0, 1, 2],
                                       [0, 2, 1],
                                       [1, 0, 2],
                                       [1, 2, 0],
                                       [2, 0, 1],
                                       [2, 1, 0]])
    rand_i = tf.random.uniform([], minval=0, maxval=6, dtype=tf.int32)
    im_swapped = tf.gather(im, channel_permutation[rand_i], axis=3)
    return im_swapped

def image_warp(image, flow, name='dense_image_warp', apply_mask=False, return_mask=False):
    x = image # tf.keras.Input(shape=(None, None, None))
    flow_in = flow #tf.keras.Input(shape=(None, None, 2))

    if return_mask or apply_mask:
        mask = tf.keras.layers.Lambda(lambda a: create_outgoing_mask(a))(flow_in)

    offset = - tf.gather(flow_in, [1,0], axis=3)
    out = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((x, offset))

    if apply_mask:
        out = out * mask

    if return_mask:
        return out, mask
    else:
        return out


def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    with tf.name_scope('create_outgoing_mask'):
        s = tf.unstack(tf.shape(flow))
        height, width, _ = s[-3::]

        grid = tf.cast(tf.meshgrid(tf.range(width), tf.range(height)), tf.float32)
        grid = tf.transpose(grid, [1,2,0])
        for i in range(len(s)-3):
            grid = tf.expand_dims(grid, 0)

        pos_x = grid[...,0] + flow[...,0]
        pos_y = grid[...,1] + flow[...,1]

        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >=  0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >=  0.0)
        inside = tf.logical_and(inside_x, inside_y)
        return tf.expand_dims(tf.cast(inside, tf.float32), -1)


def create_outgoing_mask_coor(flow_in_corr):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    with tf.name_scope('create_outgoing_mask'):
        s = tf.unstack(tf.shape(flow_in_corr))
        height, width = s[1], s[2]

        grid = tf.cast(tf.meshgrid(tf.range(width), tf.range(height)), tf.float32)
        grid = tf.transpose(grid, [1,2,0])
        grid = tf.expand_dims(grid, 0)
        grid = tf.expand_dims(grid, 3)

        pos_x = grid[...,0] + flow_in_corr[...,0]
        pos_y = grid[...,1] + flow_in_corr[...,1]

        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >=  0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >=  0.0)
        inside = tf.logical_and(inside_x, inside_y)
        return tf.cast(inside, tf.float32)


def get_epe(flow, gt):
    diff = flow - gt
    sqd = tf.square(diff)
    epe = tf.sqrt(tf.reduce_sum(sqd, axis=-1, keepdims=True))
    return epe


def create_mask(tensor, paddings):
    with tf.name_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
    return tf.stop_gradient(mask4d)