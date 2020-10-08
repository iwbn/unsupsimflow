import tensorflow as tf
import numpy as np
import cv2

def resize_bilinear(image, size):
    resized = tf.compat.v2.image.resize(image, size)
    return resized

def resize_nearest(image, size):
    resized = tf.compat.v2.image.resize(image, size,  method='nearest')
    return resized

def resize_area(image, size):
    resized = tf.compat.v2.image.resize(image, size, method='area')
    return resized

def resize_gaussian(image, size):
    resized = tf.compat.v2.image.resize(image, size, method='gaussian')
    return resized

def downsample_avg(image, scale, size):
    scale = int(scale)
    resized = tf.nn.avg_pool2d(image, scale, scale, "SAME")
    #adjusted = tf.compat.v2.image.resize(resized, size, method='bilinear')
    return resized

def get_color_map(intensity):
    s = tf.shape(intensity)
    #h, w = intensity.get_shape()[1:3]
    colormap = tf.numpy_function(get_color_map_np, [intensity], tf.float32)
    colormap = tf.reshape(colormap, [s[0], s[1], s[2], 3])
    return colormap

def get_color_map_np(intensity):
    iis = []
    for ii in intensity:
        #print (ii.shape, intensity.shape)
        res = cv2.applyColorMap(np.uint8(ii * 255.)[...,np.newaxis], cv2.COLORMAP_JET)
        res = np.float32(res) / 255.
        iis.append(res)
    iis = np.asarray(iis)
    iis = iis[:,:,:,[2,1,0]]
    return iis