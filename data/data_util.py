import tensorflow as tf
from util.image import resize_bilinear
from util.flow import resize_flow

def rand_crop_multiple(dataset, out_size, im_size=None):
    if im_size is None:
        ims = dataset.element_spec[0].shape[1:3]
    else:
        ims = im_size

    size = out_size

    _rand_func = lambda max_val: tf.random.uniform([], 0, max_val + 1, dtype=tf.int32)


    def _rand_crop(*args):
        res = []
        x_s = tf.cast(tf.shape(args[0])[1:3], tf.int32)

        rand_mh, rand_mw = x_s[0] - size[0], x_s[1] - size[1]
        rand_h, rand_w = _rand_func(rand_mh), _rand_func(rand_mw)

        for x in args:
            sy = rand_h
            sx = rand_w
            cropped = tf.image.crop_to_bounding_box(x, sy, sx, size[0], size[1])
            res.append(cropped)
        return res

    zipped = dataset
    zipped = zipped.map(_rand_crop)

    return zipped

def rand_flip_multiple(dataset, flow_indices=None):
    _rand_func = lambda: tf.random.uniform([], 0., 1., dtype=tf.float32)

    def _flow_flip(x):
        x = x[:,:,::-1]
        x = x * tf.constant([-1., 1.])[tf.newaxis, tf.newaxis, tf.newaxis]
        return x

    def _rand_flip(*args):
        p = _rand_func()
        res = []
        for i, x in enumerate(args):
            if flow_indices is not None and i in flow_indices:
                x = tf.cond(p < 0.5, lambda: x, lambda: _flow_flip(x))
                res.append(x)
            else:
                x = tf.cond(p < 0.5, lambda: x, lambda: x[:,:, ::-1])
                res.append(x)
        return res

    zipped = dataset
    zipped = zipped.map(_rand_flip)

    return zipped

def rand_resize_multiple(dataset, flow_indices=None, skip_indices=None):
    _rand_func = lambda: tf.random.uniform([], 0.9, 1.10, dtype=tf.float32)

    def _rand_resize(*args):
        p = _rand_func()
        res = []
        x_s = tf.cast(tf.shape(args[0])[1:3], tf.float32)
        x_s = x_s * p
        x_s = tf.cast(x_s, tf.int32)
        x_s = tf.reshape(x_s, [2])
        for i, x in enumerate(args):
            if skip_indices is not None and i in skip_indices:
                res.append(x)
            elif flow_indices is not None and i in flow_indices:
                x = resize_flow(x, x_s, scaling=True)
                res.append(x)
            else:
                x = resize_bilinear(x, x_s)
                res.append(x)
        return res

    zipped = dataset
    zipped = zipped.map(_rand_resize)

    return zipped


def rand_channel_swap_multiple(dataset, flow_indices=None, skip_indices=None):
    _rand_func = lambda: tf.random.uniform([], minval=0, maxval=6, dtype=tf.int32)
    channel_permutation = tf.constant([[0, 1, 2],
                                       [0, 2, 1],
                                       [1, 0, 2],
                                       [1, 2, 0],
                                       [2, 0, 1],
                                       [2, 1, 0]])

    def _rand_resize(*args):
        rand_i = _rand_func()
        res = []
        for i, x in enumerate(args):
            if skip_indices is not None and i in skip_indices:
                res.append(x)
            elif flow_indices is not None and i in flow_indices:
                res.append(x)
            else:
                x = tf.gather(x, channel_permutation[rand_i], axis=3)
                res.append(x)
        return res

    zipped = dataset
    zipped = zipped.map(_rand_resize)

    return zipped


def crop_or_pad_multiple(dataset, size, skip_indices=None):

    def _rand_resize(*args):
        res = []
        for i, x in enumerate(args):
            if skip_indices is not None and i in skip_indices:
                res.append(x)
            else:
                x_ = tf.image.resize_with_crop_or_pad(x, size[0], size[1])
                res.append(x_)
        return res

    zipped = dataset
    zipped = zipped.map(_rand_resize)

    return zipped

def distort_color_multiple(dataset, skip_indices=None):

    def _distort_color(*args):
        res = []
        for i, x in enumerate(args):
            if skip_indices is not None and i in skip_indices:
                res.append(x)
            else:
                x_ = distort_color(x)
                res.append(x_)
        return res

    zipped = dataset
    zipped = zipped.map(_distort_color)

    return zipped

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope('distort_color'):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return image

def convert_elems_to_dict(dataset, keys):
    dataset = dataset.map(lambda *args: {keys[i]: args[i] for i in range(len(keys))})
    return dataset

def convert_feat_label_to_dict(dataset, keys):
    assert len(keys) == 2
    assert type(keys[0]) == list
    assert type(keys[1]) == list

    def _process(*args):
        tr = {}
        ts = {}
        cnt = 0
        for key in keys[0]:
            tr[key] = args[cnt]
            cnt += 1

        for key in keys[1]:
            ts[key] = args[cnt]
            cnt += 1

        return [tr, ts]

    dataset = dataset.map(_process)
    return dataset


def overlapping_blocker(tensor, block_size=2, stride=1):
    tensor_idx = tf.range(tf.shape(tensor)[0])
    selected_indices =  tf.squeeze(tf.image.extract_patches(tensor_idx[None,...,None, None],
                                                            sizes=[1, block_size, 1, 1],
                                                            strides=[1, stride, 1, 1],
                                                            rates=[1, 1, 1, 1], padding='VALID'),
                                   axis=[0,2])
    res = tf.gather(tensor, selected_indices)
    out_shape = tf.concat([tf.shape(selected_indices), tf.shape(tensor)[1:]], axis=0)
    res = tf.reshape(res, out_shape)
    return res