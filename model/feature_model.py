import tensorflow as tf
import os
import numpy as np
from model.pwcnet import PWCNet
from model_base import Model
from util.flow import resize_flow, image_warp
from util.feature import ternary_transform
from util.image import resize_nearest, resize_bilinear
from tqdm import tqdm
from model.similarity import get_fused_similarity
from data.flow_datasets import input_fn
from box import Box
from collections.abc import Iterable

CENSUS_PATCH = 7

class FeatureModel(Model):
    def model_def(self):
        params = self.params
        x = tf.keras.Input(shape=(None, None, None, 3))

        if 'backward_flow' in params.keys() and params.backward_flow:
            pwc = PWCNet(params, backward_flow=True)
        else:
            pwc = PWCNet(params, backward_flow=False)

        model = pwc.get_model_fn(x)

        im1_convs = pwc.im1_convs
        im2_convs = pwc.im2_convs

        pred_flows = pwc.decoder_fw_res.mid_pred_flows
        pred_flows = pred_flows[0:-1] + [pwc.decoder_fw_res.pred_flow]
        pred_flows = [f * pwc.flow_scale for f in pred_flows]
        pre_context_flow = pwc.decoder_fw_res.mid_pred_flows[-1] * pwc.flow_scale

        get_scales = tf.keras.layers.Lambda(
            lambda x: tf.convert_to_tensor([max(float(2 ** (6 - i)), 4.) for i in range(len(pred_flows))])
        )
        scales = get_scales(x)

        outputs = {'im1': x[:, 0], 'im2': x[:, 1],
                   'flows_fw': pred_flows,
                   'pre_context_flow_fw': pre_context_flow,
                   'im1_convs': im1_convs,
                   'im2_convs': im2_convs,
                   'flow_fw_full': resize_flow(pred_flows[-1], tf.shape(x)[2:4], scaling=False),
                   'flow_scales': scales,
                   }

        if 'backward_flow' in params.keys() and params.backward_flow:
            pred_flows_bw = pwc.decoder_bw_res.mid_pred_flows
            pred_flows_bw = pred_flows_bw[0:-1] + [pwc.decoder_bw_res.pred_flow]
            pred_flows_bw = [f * pwc.flow_scale for f in pred_flows_bw]
            outputs['flows_bw'] = pred_flows_bw
            outputs['flow_bw_full'] = resize_flow(pred_flows_bw[-1], tf.shape(x)[2:4], scaling=False)
            outputs['pre_context_flow_bw'] = pwc.decoder_bw_res.mid_pred_flows[-1] * pwc.flow_scale

        self.pwc = pwc
        self.pwc_model = model

        wrapper = tf.keras.Model(inputs=x, outputs=outputs)

        return wrapper

    def init_train_dataset(self):
        dataset_params = self.params.copy()
        dataset_params['dataset'] = self.params.train_dataset
        dataset = input_fn("train", dataset_params)
        dataset = dataset.shuffle(500 * self.params.batch_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.params.batch_size)
        dataset = dataset.prefetch(2)

        return dataset

    def init_val_dataset(self):
        dataset_params = self.params.copy()
        dataset_params['dataset'] = self.params.val_dataset
        dataset = input_fn("val", dataset_params)
        dataset = dataset.shuffle(500 * self.params.batch_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.params.batch_size)
        dataset = dataset.prefetch(2)

        return dataset

    def loss_feature_separation(self, fused_sim, mask):
        noc_mask = mask
        occ_mask = (1. - mask)

        noc_mean = tf.reduce_sum(fused_sim * noc_mask, [1, 2], keepdims=True) / (
                    tf.reduce_sum(noc_mask, [1, 2], keepdims=True) + 1e-3)
        occ_mean = tf.reduce_sum(fused_sim * occ_mask, [1, 2], keepdims=True) / (
                    tf.reduce_sum(occ_mask, [1, 2], keepdims=True) + 1e-3)

        k = tf.stop_gradient((occ_mean + noc_mean)/2.)

        quad_loss = - tf.square(fused_sim - k)
        quad_loss = tf.reduce_mean(quad_loss)

        return quad_loss

    def loss_reg_census(self, t1, t2_warped, mask, fused_sim):
        dist = _hamming_distance(t1, t2_warped) * fused_sim
        loss = abs_robust_loss(dist, mask, mask_mean=False)

        return loss

    def loss_cond_smooth(self, smooth, fused_sim, mask):
        noc_mask = mask
        occ_mask = (1. - mask)

        noc_mean = tf.reduce_sum(fused_sim * noc_mask, [1, 2], keepdims=True) / (
                tf.reduce_sum(noc_mask, [1, 2], keepdims=True) + 1e-3)
        occ_mean = tf.reduce_sum(fused_sim * occ_mask, [1, 2], keepdims=True) / (
                tf.reduce_sum(occ_mask, [1, 2], keepdims=True) + 1e-3)

        k = tf.stop_gradient((occ_mean + noc_mean))

        loss = tf.where(fused_sim < k, smooth, tf.zeros_like(fused_sim))
        loss = tf.reduce_mean(loss)

        return loss

    def loss_dd(self, teacher_cropped, student_cropped, tea_mask, stu_mask):
        dd_mask_fw = tf.clip_by_value(tea_mask - stu_mask, 0., 1.)

        dd_loss = abs_robust_loss(tf.stop_gradient(teacher_cropped) - student_cropped, dd_mask_fw)
        return dd_loss

    def calculate_losses(self, inputs, model_outputs, params):
        assert params.backward_flow == True

        losses_outputs = {}

        im2_warped = image_warp(inputs['im'][:,1], model_outputs["flow_fw_full"])
        im1_warped = image_warp(inputs['im'][:,0], model_outputs["flow_bw_full"])

        losses_outputs['im2_warped'] = im2_warped
        losses_outputs['im1_warped'] = im1_warped

        mask_fw, mask_bw, _, _ = get_mask(model_outputs["flow_fw_full"], model_outputs["flow_bw_full"])

        losses_outputs['mask_fw_full'] = mask_fw
        losses_outputs['mask_bw_full'] = mask_bw

        if params.exist_or('pyramid_photo_weight'):
            losses_outputs['im2_warped_pyramid'] = []
            losses_outputs['im1_warped_pyramid'] = []
            for ff, fb in zip(model_outputs['flows_fw'], model_outputs['flows_bw']):
                flow_fw_resized = resize_bilinear(ff, tf.shape(inputs['im'][:, 0])[1:3])
                flow_bw_resized = resize_bilinear(fb, tf.shape(inputs['im'][:, 1])[1:3])

                im2_s_warped = image_warp(inputs['im'][:, 1], flow_fw_resized, apply_mask=True)
                im1_s_warped = image_warp(inputs['im'][:, 0], flow_bw_resized, apply_mask=True)

                losses_outputs['im2_warped_pyramid'].append(im2_s_warped)
                losses_outputs['im1_warped_pyramid'].append(im1_s_warped)

        if params.exist_or('census_weight', 'reg_census_weight'):
            t1 = ternary_transform(inputs['im'][:, 0], CENSUS_PATCH)
            t2 = ternary_transform(inputs['im'][:, 1], CENSUS_PATCH)
            t2_warped = ternary_transform(im2_warped, CENSUS_PATCH)
            t1_warped = ternary_transform(im1_warped, CENSUS_PATCH)

            losses_outputs['t2_warped'] = t2_warped
            losses_outputs['t1_warped'] = t1_warped

        if params.exist_or('feat_sep_weight', 'reg_census_weight', 'cond_smooth_weight'):
            # simmap calculation
            fused_sim_fw = get_fused_similarity(model_outputs, params, 'im1', 'im2', fw_str='fw', feat_str='convs')
            fused_sim_bw = get_fused_similarity(model_outputs, params, 'im2', 'im1', fw_str='bw', feat_str='convs')

            losses_outputs['fused_sim_fw_res'] = fused_sim_fw
            losses_outputs['fused_sim_bw_res'] = fused_sim_bw

            fb_mask_sim_fw = resize_nearest(losses_outputs['mask_fw_full'], tf.shape(fused_sim_fw['fused_prod'])[1:3])
            fb_mask_sim_bw = resize_nearest(losses_outputs['mask_bw_full'], tf.shape(fused_sim_bw['fused_prod'])[1:3])

            losses_outputs['fb_mask_sim_fw'] = fb_mask_sim_fw
            losses_outputs['fb_mask_sim_bw'] = fb_mask_sim_bw

        if params.exist_or('reg_census_weight'):
            losses_outputs['feat_sim_mask_fw'] = tf.stop_gradient(losses_outputs['fused_sim_fw_res']['fused_prod_full'])
            losses_outputs['feat_sim_mask_bw'] = tf.stop_gradient(losses_outputs['fused_sim_bw_res']['fused_prod_full'])


        if params.exist_or('cond_smooth_weight'):
            losses_outputs['flow_sim_resized_fw'] = resize_bilinear(model_outputs["flow_fw_full"],
                                                                    tf.shape(fused_sim_fw['fused_prod'])[1:3]) / 2.
            losses_outputs['flow_sim_resized_bw'] = resize_bilinear(model_outputs["flow_bw_full"],
                                                                    tf.shape(fused_sim_bw['fused_prod'])[1:3]) / 2.

        # DD preperation step
        if params.get_or_none('dd_weight'):
            ims = inputs['im']
            DD_PADDING = params.dd_padding
            offset_h = tf.random.uniform([], -DD_PADDING, DD_PADDING, dtype=tf.int32)
            offset_w = tf.random.uniform([], -DD_PADDING, DD_PADDING, dtype=tf.int32)
            ims_crop = ims[:, :, DD_PADDING + offset_h:-DD_PADDING + offset_h,
                       DD_PADDING + offset_w:-DD_PADDING + offset_w]
            cropped_outputs = self.model(ims_crop - 0.5)

            tea_mask_fw, tea_mask_bw, _, _ = get_mask(inputs['teacher_flow_fw_full'][:,0], inputs['teacher_flow_bw_full'][:,0])
            stu_mask_fw, stu_mask_bw, _, _ = get_mask(cropped_outputs['flow_fw_full'], cropped_outputs['flow_bw_full'])

            tea_mask_fw_crop = tea_mask_fw[:, DD_PADDING + offset_h:-DD_PADDING + offset_h,
                                              DD_PADDING + offset_w:-DD_PADDING + offset_w]
            tea_mask_bw_crop = tea_mask_bw[:, DD_PADDING + offset_h:-DD_PADDING + offset_h,
                                              DD_PADDING + offset_w:-DD_PADDING + offset_w]

            tea_flow_fw_crop = inputs['teacher_flow_fw_full'][:,0][:, DD_PADDING + offset_h:-DD_PADDING + offset_h,
                                                                 DD_PADDING + offset_w:-DD_PADDING + offset_w]
            tea_flow_bw_crop = inputs['teacher_flow_bw_full'][:,0][:, DD_PADDING + offset_h:-DD_PADDING + offset_h,
                                                                 DD_PADDING + offset_w:-DD_PADDING + offset_w]

            losses_outputs['dd_cropped_outputs'] = cropped_outputs
            losses_outputs['dd_offset_h'] = offset_h
            losses_outputs['dd_offset_w'] = offset_w
            losses_outputs['teacher_mask_fw'] = tea_mask_fw
            losses_outputs['teacher_mask_bw'] = tea_mask_bw
            losses_outputs['teacher_mask_fw_crop'] = tea_mask_fw_crop
            losses_outputs['teacher_mask_bw_crop'] = tea_mask_bw_crop
            losses_outputs['student_mask_fw'] = stu_mask_fw
            losses_outputs['student_mask_bw'] = stu_mask_bw
            losses_outputs['teacher_flow_fw_crop'] = tea_flow_fw_crop
            losses_outputs['teacher_flow_bw_crop'] = tea_flow_bw_crop

        losses = {}
        if params.exist_or('photo_weight'):
            if params.use_occ:
                mask_fw_full = losses_outputs['mask_fw_full']
                mask_bw_full = losses_outputs['mask_bw_full']
            else:
                mask_fw_full = tf.ones_like(losses_outputs['mask_fw_full'])
                mask_bw_full = tf.ones_like(losses_outputs['mask_bw_full'])

            if params.photo_weight > 0.:
                im_diff_fw = inputs['im'][:, 0] - im2_warped
                im_diff_bw = inputs['im'][:, 1] - im1_warped
                losses['photo_fw'] = abs_robust_loss(im_diff_fw, mask_fw_full)
                losses['photo_bw'] = abs_robust_loss(im_diff_bw, mask_bw_full)

        if params.exist_or('pyramid_photo_weight'):
            if params.use_occ:
                mask_fw_full = losses_outputs['mask_fw_full']
                mask_bw_full = losses_outputs['mask_bw_full']
            else:
                mask_fw_full = tf.ones_like(losses_outputs['mask_fw_full'])
                mask_bw_full = tf.ones_like(losses_outputs['mask_bw_full'])

            if params.exist_or('pyramid_layer_weights'):
                pyramid_layer_weights = params['pyramid_layer_weights']
            else:
                pyramid_layer_weights = [1.] * len(losses_outputs['im2_warped_pyramid'])

            pyramid_photo_fw = []
            pyramid_photo_bw = []

            for im2_warped_t, im1_warped_t, weight in zip(losses_outputs['im2_warped_pyramid'],
                                                          losses_outputs['im1_warped_pyramid'],
                                                          pyramid_layer_weights):
                im_diff_fw = inputs['im'][:, 0] - im2_warped_t
                im_diff_bw = inputs['im'][:, 1] - im1_warped_t

                fw_loss = abs_robust_loss(im_diff_fw, mask_fw_full)
                bw_loss = abs_robust_loss(im_diff_bw, mask_bw_full)

                pyramid_photo_fw.append(fw_loss * weight)
                pyramid_photo_bw.append(bw_loss * weight)

            losses['pyramid_photo_fw'] = tf.add_n(pyramid_photo_fw)
            losses['pyramid_photo_bw'] = tf.add_n(pyramid_photo_bw)

        if params.get_or_zero('census_weight') > 0.:
            if params.use_occ:
                mask_fw_full = losses_outputs['mask_fw_full']
                mask_bw_full = losses_outputs['mask_bw_full']
            else:
                mask_fw_full = tf.ones_like(losses_outputs['mask_fw_full'])
                mask_bw_full = tf.ones_like(losses_outputs['mask_bw_full'])
            losses['census_fw'] = abs_robust_loss(_hamming_distance(t1, t2_warped), mask_fw_full, mask_mean=False)
            losses['census_bw'] = abs_robust_loss(_hamming_distance(t2, t1_warped), mask_bw_full, mask_mean=False)

        if params.get_or_zero('reg_census_weight') > 0.:
            if params.use_occ:
                mask_fw_full = losses_outputs['mask_fw_full']
                mask_bw_full = losses_outputs['mask_bw_full']
            else:
                mask_fw_full = tf.ones_like(losses_outputs['mask_fw_full'])
                mask_bw_full = tf.ones_like(losses_outputs['mask_bw_full'])
            losses['reg_census_fw'] = abs_robust_loss(_hamming_distance(t1, t2_warped),
                                                  mask_fw_full * losses_outputs['feat_sim_mask_fw'], mask_mean=False)
            losses['reg_census_bw'] = abs_robust_loss(_hamming_distance(t2, t1_warped),
                                                  mask_bw_full * losses_outputs['feat_sim_mask_bw'], mask_mean=False)

        if params.get_or_zero('cond_smooth_weight') > 0.:
            smooth_fw = smooth_1st_sp(losses_outputs['flow_sim_resized_fw'])
            smooth_bw = smooth_1st_sp(losses_outputs['flow_sim_resized_bw'])
            losses['cond_smooth_fw'] = self.loss_cond_smooth(smooth_fw, losses_outputs['fused_sim_fw_res']['fused_prod'],
                                                        losses_outputs['fb_mask_sim_fw'])
            losses['cond_smooth_bw'] = self.loss_cond_smooth(smooth_bw, losses_outputs['fused_sim_bw_res']['fused_prod'],
                                                        losses_outputs['fb_mask_sim_bw'])

        if params.get_or_zero('feat_sep_weight') > 0.:
            losses['feat_sep_fw'] = self.loss_feature_separation(losses_outputs['fused_sim_fw_res']['fused_prod'],
                                                                    losses_outputs['fb_mask_sim_fw'])
            losses['feat_sep_bw'] = self.loss_feature_separation(losses_outputs['fused_sim_bw_res']['fused_prod'],
                                                                    losses_outputs['fb_mask_sim_bw'])

        if params.get_or_zero('dd_weight') > 0.:
            losses['dd_fw'] = self.loss_dd(losses_outputs['teacher_flow_fw_crop'],
                                           losses_outputs['dd_cropped_outputs']['flow_fw_full'],
                                           losses_outputs['teacher_mask_fw_crop'],
                                           losses_outputs['student_mask_fw'])
            losses['dd_bw'] = self.loss_dd(losses_outputs['teacher_flow_bw_crop'],
                                           losses_outputs['dd_cropped_outputs']['flow_bw_full'],
                                           losses_outputs['teacher_mask_bw_crop'],
                                           losses_outputs['student_mask_bw'])

        loss_vals = []
        for k, v in losses.items():
            weight_key = k[:-3] + "_weight"
            loss_vals.append(v * params[weight_key])
            print("loss %s added with weight %f" % (k, params[weight_key]))

        if len(loss_vals) > 0:
            loss = tf.add_n(loss_vals)
        else:
            loss = tf.zeros([], dtype=tf.float32)

        losses_outputs['losses'] = losses

        return loss, losses_outputs


    @tf.function
    def feed_step(self, inputs):
        x, y = inputs
        net_inputs = x.copy()
        for k, v in y.items():
            net_inputs[k] = v
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.model(x['im'] - 0.5)
            loss, losses_outputs = self.calculate_losses(net_inputs, outputs, self.params)

        outputs['losses_outputs'] = losses_outputs
        outputs['loss'] = loss

        return outputs

    @tf.function
    def training_step(self, inputs):
        x, y = inputs
        net_inputs = x.copy()
        for k, v in y.items():
            net_inputs[k] = v
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.model(x['im'] - 0.5)
            loss, losses_outputs = self.calculate_losses(net_inputs, outputs, self.params)

        outputs['losses_outputs'] = losses_outputs
        outputs['loss'] = loss

        trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        grad_vars = list(zip(gradients, trainable_variables))
        self.optimizer.apply_gradients(grad_vars)

        return outputs

    def initialize(self):
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
                                                                 self.params.lr_decay_steps,
                                                                 0.5,
                                                                 staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        s = self.obj_to_save

        s['optimizer'] = self.optimizer
        s['model'] = self.model

        self.train_dataset = self.init_train_dataset()
        self.val_dataset = self.init_val_dataset()

        if self.params.get_or_none('pretrained_model_v2_ckpt'):
            self.pwc_model.load_weights(self.params['pretrained_model_v2_ckpt'])

        ckpt_tmp = tf.train.Checkpoint(model=self.model)
        if self.params.get_or_none('pretrained_ckpt'):
            ckpt_tmp.restore(self.params['pretrained_ckpt'])

        self.writer = tf.summary.create_file_writer(self.params['ckpt_path'])

        val_path = os.path.join(self.params['ckpt_path'], "val")
        os.makedirs(val_path, exist_ok=True)
        self.val_writer = tf.summary.create_file_writer(val_path)

    def validate_with_dataset(self, dataset):
        @tf.function
        def _train_step(inputs):
            x, y = inputs
            net_inputs = x.copy()
            for k, v in y.items():
                net_inputs[k] = v
            params = self.params.copy()
            if params.exist_or("dd_weight"):
                del params['dd_weight']
            with tf.GradientTape(persistent=True) as tape:
                outputs = self.model(x['im'] - 0.5)
                loss, losses_outputs = self.calculate_losses(net_inputs, outputs, params)

            outputs['losses_outputs'] = losses_outputs
            outputs['loss'] = loss
            return outputs

        metrics = {
            'epe': tf.keras.metrics.Mean(),
        }
        len_dataset = tf.data.experimental.cardinality(dataset)
        diter = iter(dataset)
        pbar = tqdm(range(len_dataset))
        pbar.set_description("Validating...")
        for i, b in zip(pbar, diter):
            outputs = _train_step(b)
            losses = outputs['losses_outputs']['losses']
            for k, loss in losses.items():
                metrics.setdefault(k, tf.keras.metrics.Mean()).update_state(loss)
            epes = self.get_epe(outputs['flow_fw_full'], b[1]['flow'][:, 0])
            metrics['epe'].update_state(epes)
        res = {k: v.result() for k, v in metrics.items()}

        return res

    def get_epe(self, pred_flow, gt_flow):
        sqer = tf.square(pred_flow - gt_flow)
        sqer = tf.reduce_sum(sqer, axis=-1, keepdims=True)
        epes = tf.sqrt(sqer)
        epes = tf.reduce_mean(epes, [1,2,3])
        return epes

    def train(self):
        train_dataset = self.train_dataset.repeat(-1)
        val_dataset = self.val_dataset

        step = self.global_step.numpy()

        def _validate():
            with self.val_writer.as_default():
                res = self.validate_with_dataset(val_dataset)
                for k, v in res.items():
                    tf.summary.scalar("summary/%s" % k, v, step=self.global_step.numpy())

        if step == 0:
            self.ckpt_manager.save(self.global_step.numpy())
            # validate
            #_validate()


        train_iter = iter(train_dataset)
        while self.global_step.numpy() < self.params['max_steps']:
            with self.writer.as_default():
                pbar = tqdm(range(self.params['val_steps']))
                lm = Box()
                lm.epe = tf.keras.metrics.Mean()

                for cnt in pbar:
                    batch = next(train_iter)
                    outputs = self.training_step(batch)

                    epes = self.get_epe(outputs['flow_fw_full'], batch[1]['flow'][:, 0])
                    lm.epe.update_state(epes)

                    for k, loss in outputs['losses_outputs']['losses'].items():
                        lm.setdefault(k, tf.keras.metrics.Mean()).update_state(loss)

                    if cnt % 100 == 0:
                        logs = []
                        for k in ['epe']:
                            logs.append("%s=%.4f" % (k, lm[k].result().numpy()))
                        pbar.set_description("[step %d] " % self.global_step.numpy() + " | ".join(logs))

                        for k, v in lm.items():
                            tf.summary.scalar("summary/%s" % k, v.result(), step=self.global_step.numpy())

                        tf.summary.scalar("summary/lr", self.lr(self.global_step), step=self.global_step.numpy())


                        self.writer.flush()

                    # increment global step
                    self.global_step.assign_add(1)

            _validate()
            # save
            self.ckpt_manager.save(self.global_step.numpy())


def get_mask(flow_fw, flow_bw, border_mask=None):
    if border_mask is None:
        bmask_fw = create_outgoing_mask(flow_fw)
        bmask_bw = create_outgoing_mask(flow_bw)
    else:
        bmask_fw = border_mask
        bmask_bw = border_mask

    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    fb_occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh, tf.float32)
    fb_occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh, tf.float32)

    mask_fw = bmask_fw * (1 - fb_occ_fw)
    mask_bw = bmask_bw * (1 - fb_occ_bw)

    return mask_fw, mask_bw, bmask_fw, bmask_bw


def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)


def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    with tf.name_scope('create_outgoing_mask'):
        num_batch, height, width, _ = tf.unstack(tf.shape(flow))

        grid = tf.cast(tf.meshgrid(tf.range(width), tf.range(height)), tf.float32)
        grid = tf.transpose(grid, [1,2,0])
        grid = tf.expand_dims(grid, 0)
        pos_x = grid[...,0] + flow[...,0]
        pos_y = grid[...,1] + flow[...,1]

        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >=  0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >=  0.0)
        inside = tf.logical_and(inside_x, inside_y)
        return tf.expand_dims(tf.cast(inside, tf.float32), 3)


def _hamming_distance(t1, t2):
    dist = tf.square(t1 - t2)
    dist_norm = dist / (0.1 + dist)
    dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
    return dist_sum

# from ddflow
def abs_robust_loss(diff, mask, q=0.4, mask_mean=False):

    diff = tf.pow((tf.abs(diff)+0.01), q)
    diff = tf.multiply(diff, mask)
    if mask_mean:
        loss = tf.reduce_sum(diff) / (tf.reduce_sum(mask) + 1e-4)
    else:
        loss = tf.reduce_mean(tf.reduce_sum(diff, axis=(3)))

    return loss

def smooth_1st_sp(flow):
    grad = tf.image.image_gradients(flow)
    grad = tf.transpose(grad, [1,2,3,4,0])
    grad_sq = tf.square(grad)
    grad_sq = tf.reduce_sum(grad_sq, axis=[3,4], keepdims=False)[...,tf.newaxis]
    return grad_sq


if __name__ == "__main__":
    from conf.conf import RGBOcc, PhotoBaseline, CensusOcc, CensusOccDD, Ours, OursFinal
    import os
    #tf.config.experimental_run_functions_eagerly(True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('WARNING')
    tf.autograph.set_verbosity(0)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if False and gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=13000)])
        except RuntimeError as e:
            print(e)

    Params = OursFinal
    params = Params()
    params.ckpt_path = "/mnt/data/wbim/new_ckpt/ours_final_01"
    params.save_or_load()


    model = FeatureModel(params)

    model.train()