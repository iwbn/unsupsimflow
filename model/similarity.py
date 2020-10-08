import tensorflow as tf
from util.image import downsample_avg, resize_nearest
from util.flow import image_warp


def get_fused_similarity(network_outputs, params, im1_str='im1', im2_str='im2', fw_str='fw', feat_str='convs'):
    im1 = network_outputs[im1_str]
    im1_convs = network_outputs[im1_str + "_" + feat_str]
    im2_convs = network_outputs[im2_str + "_" + feat_str]

    im1_convs = list(reversed(im1_convs))
    im2_convs = list(reversed(im2_convs))

    final_shape = tf.shape(im1)
    final_feat_shape = tf.shape(im1_convs[-1])

    feat_masks_fw = []
    feat_dists_fw = []
    feat_sims_fw = []
    for feat_level in range(len(im1_convs)):
        feat1l = im1_convs[feat_level]
        feat2l = im2_convs[feat_level]

        feat1l = tf.stop_gradient(feat1l)
        feat2l = tf.stop_gradient(feat2l)

        feat1ln = tf.nn.l2_normalize(feat1l, axis=3)

        feat_scale = 2 ** (6-feat_level)

        flow_fw_full = network_outputs['flow_' + fw_str + '_full']

        flow_fw_resized = downsample_avg(flow_fw_full, feat_scale, tf.shape(feat1l)[1:3]) / feat_scale
        #print(feat_scale, flow_fw_resized.shape, feat2l.shape)
        feat2_warped, out_mask_fw = image_warp(feat2l, flow_fw_resized, apply_mask=True, return_mask=True)
        feat2_warped = tf.nn.l2_normalize(feat2_warped, axis=3)

        cos_sim_fw = tf.reduce_sum(feat1ln * feat2_warped, axis=-1, keepdims=True)

        feat_masks_fw.append(resize_nearest(out_mask_fw, final_feat_shape[1:3]))

        feat_sim_fw = (1. + cos_sim_fw) / 2.

        #feat_sim_fw = tf.clip_by_value(feat_sim_fw + 1. - out_mask_fw, 0., 1.)

        feat_dist_fw = 1. - feat_sim_fw

        feat_dist_fw = resize_nearest(feat_dist_fw, final_feat_shape[1:3])
        feat_sim_fw = resize_nearest(feat_sim_fw, final_feat_shape[1:3])

        feat_dists_fw.append(feat_dist_fw)
        feat_sims_fw.append(feat_sim_fw)

    fused_prod_fw = tf.reduce_prod(tf.stack(feat_sims_fw), axis=0)
    fused_mean_fw = tf.reduce_mean(tf.stack(feat_sims_fw), axis=0)
    fused_mask =tf.reduce_prod(tf.stack(feat_masks_fw), axis=0)

    fused_prod_fw_full = resize_nearest(fused_prod_fw, final_shape[1:3])
    fused_mean_fw_full = resize_nearest(fused_mean_fw, final_shape[1:3])


    similarity_maps = {
        'fused_prod': fused_prod_fw,
        'fused_mean': fused_mean_fw,
        'fused_mask': fused_mask,
        'fused_prod_full': fused_prod_fw_full,
        'fused_mean_full': fused_mean_fw_full
    }

    return similarity_maps