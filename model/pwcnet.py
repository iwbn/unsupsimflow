import tensorflow as tf
from util.feature import Correlation
import numpy as np
from util.flow import image_warp
from util.flow import resize_flow
from box import Box

class PWCNet():
    def __init__(self, params, backward_flow=False, **kwargs):
        #super(PWCNet, self).__init__(**kwargs)
        self.data_format = 'channels_last'
        self.activation = tf.keras.layers.LeakyReLU(0.1)
        self.flow_scale = 1.0
        self.backward_flow = backward_flow
        self.weight_decay = params.weight_decay

        self.kernel_initializer = tf.keras.initializers.HeNormal()

        self._Conv = lambda dim, ksize, stride, name: tf.keras.layers.Conv2D(dim, ksize, stride, 'same',
                                                                        activation=self.activation,
                                                                        data_format=self.data_format,
                                                                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                                        kernel_initializer=self.kernel_initializer,
                                                                        name=name)

        self._Conv_na = lambda dim, ksize, stride, name: tf.keras.layers.Conv2D(dim, ksize, stride, 'same',
                                                                             activation=None,
                                                                             data_format=self.data_format,
                                                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                                                 self.weight_decay),
                                                                             kernel_initializer=self.kernel_initializer,
                                                                             name=name)

        self._ConvDilated = lambda dim, ksize, stride, rate, name: tf.keras.layers.Conv2D(dim, ksize, stride, 'same',
                                                                             activation=self.activation,
                                                                             data_format=self.data_format,
                                                                             dilation_rate=rate,
                                                                             kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                                             kernel_initializer=self.kernel_initializer,
                                                                             name=name)

        self.max_displacement = 4
        self.corr_rate = 1

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()


    def get_encoder(self):
        x = tf.keras.layers.Input(shape=(None, None, 3))
        net = x
        net = self._Conv(16, 4, 2, 'conv1a')(net)
        net = self._Conv(16, 3, 1, 'conv1aa')(net)
        net = self._Conv(16, 3, 1, 'conv1b')(net)
        conv1 = net

        net = self._Conv(32, 4, 2, 'conv2a')(net)
        net = self._Conv(32, 3, 1, 'conv2aa')(net)
        net = self._Conv(32, 3, 1, 'conv2b')(net)
        conv2 = net

        net = self._Conv(64, 4, 2, 'conv3a')(net)
        net = self._Conv(64, 3, 1, 'conv3aa')(net)
        net = self._Conv(64, 3, 1, 'conv3b')(net)
        conv3 = net

        net = self._Conv(96, 4, 2, 'conv4a')(net)
        net = self._Conv(96, 3, 1, 'conv4aa')(net)
        net = self._Conv(96, 3, 1, 'conv4b')(net)
        conv4 = net

        net = self._Conv(128, 4, 2, 'conv5a')(net)
        net = self._Conv(128, 3, 1, 'conv5aa')(net)
        net = self._Conv(128, 3, 1, 'conv5b')(net)
        conv5 = net

        net = self._Conv(196, 4, 2, 'conv6a')(net)
        net = self._Conv(196, 3, 1, 'conv6aa')(net)
        net = self._Conv(196, 3, 1, 'conv6b')(net)
        conv6 = net

        return tf.keras.Model(x, [conv1, conv2, conv3, conv4, conv5, conv6], name='encoder')

    def get_decoder(self, name="decoder"):
        c1 = tf.keras.layers.Input(shape=(2, None, None, 16), name='conv1')
        c2 = tf.keras.layers.Input(shape=(2, None, None, 32), name='conv2')
        c3 = tf.keras.layers.Input(shape=(2, None, None, 64), name='conv3')
        c4 = tf.keras.layers.Input(shape=(2, None, None, 96), name='conv4')
        c5 = tf.keras.layers.Input(shape=(2, None, None, 128), name='conv5')
        c6 = tf.keras.layers.Input(shape=(2, None, None, 196), name='conv6')

        cs = [c1, c2, c3, c4, c5, c6]

        cs_r = list(reversed(cs))

        up_flow = None
        up_feat = None
        self.dec_nets = []
        pred_flows = []
        for i in range(len(cs_r) - 1):
            c = cs_r[i]
            cname = c.name[:-2]
            scale = 2.0 ** (len(cs_r) - i)
            corr = Correlation(self.max_displacement, self.corr_rate, name="%s_corr" % cname)
            c_1, c_2 = c[:, 0], c[:, 1]
            if up_flow is not None:
                in_shape = tf.shape(c_1)[1:3]
                up_flow = up_flow[:, 0:in_shape[0], 0:in_shape[1], :]
                up_flow = tf.reshape(up_flow, [-1, in_shape[0], in_shape[1], 2])
                up_feat = up_feat[:, 0:in_shape[0], 0:in_shape[1], :]
                c_2 = image_warp(c_2, up_flow * self.flow_scale / scale, name='%s_warp' % cname, apply_mask=False)

            cv = corr([c_1, c_2])
            fused = self.activation(cv)
            if up_flow is not None:
                fused = tf.concat([fused, c_1, up_flow, up_feat], axis=3)

            is_last = (i == len(cs_r) - 2)
            dec_net = self.get_dec_net(fused.get_shape()[-1], not is_last, "%s_up" % (cname))
            self.dec_nets.append(dec_net)
            if is_last:
                net, pred_flow = dec_net(fused)
            else:
                _, pred_flow, up_flow, up_feat = dec_net(fused)

            pred_flows.append(pred_flow)

        self.context_net = self.get_context_net(net.get_shape()[-1], name="context_net")
        pred_flow = pred_flow + self.context_net(net)
        results = Box(
            {
                "mid_pred_flows": pred_flows,
                "pred_flow": pred_flow
            }
        )

        return tf.keras.Model([c1, c2, c3, c4, c5, c6], results, name=name)

    def get_dec_net(self, input_dim, return_upflow=True, name="decoder_net"):
        x = tf.keras.layers.Input(shape=(None, None, input_dim))
        net = x
        tnet = self._Conv(128, 3, 1, 'conv1')(net)
        net = tf.concat([tnet, net], axis=3, name="cat1")
        tnet = self._Conv(128, 3, 1, 'conv2')(net)
        net = tf.concat([tnet, net], axis=3, name="cat2")
        tnet = self._Conv(96, 3, 1, 'conv3')(net)
        net = tf.concat([tnet, net], axis=3, name="cat3")
        tnet = self._Conv(64, 3, 1, 'conv4')(net)
        net = tf.concat([tnet, net], axis=3, name="cat4")
        tnet = self._Conv(32, 3, 1, 'conv5')(net)
        net = tf.concat([tnet, net], axis=3, name="cat5")

        pred_flow = self._Conv_na(2, 3, 1, 'pred_flow')(net)

        if return_upflow:
            up_flow = tf.keras.layers.Conv2DTranspose(2, 4, 2, "same", name="up_flow",
                                                      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                      kernel_initializer=self.kernel_initializer)(pred_flow)
            up_feat = tf.keras.layers.Conv2DTranspose(2, 4, 2, "same", name="up_feat",
                                                      kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                      kernel_initializer=self.kernel_initializer)(net)

            return tf.keras.Model(x, [net, pred_flow, up_flow, up_feat], name=name)
        else:
            return tf.keras.Model(x, [net, pred_flow], name=name)

    def get_context_net(self, input_dim, name="context_net"):
        x = tf.keras.layers.Input(shape=(None, None, input_dim))
        net = x
        net = self._ConvDilated(128, 3, 1, 1, "conv1")(net)
        net = self._ConvDilated(128, 3, 1, 2, "conv2")(net)
        net = self._ConvDilated(128, 3, 1, 4, "conv3")(net)
        net = self._ConvDilated(96, 3, 1, 8, "conv4")(net)
        net = self._ConvDilated(64, 3, 1, 16, "conv5")(net)
        net = self._ConvDilated(32, 3, 1, 1, "conv6")(net)
        flow = self._Conv_na(2, 3, 1, 'pred_flow')(net)

        return tf.keras.Model(x, flow, name=name)

    def configure_convs(self, im1_convs, im2_convs, names):
        convs = []
        for s1, s2, n in zip(im1_convs, im2_convs, names):
            convs.append(tf.stack([s1, s2], axis=1, name="stack_convs_%d" % n))
        return convs

    #@tf.function
    def get_model_fn(self, x):
        im1 = x[:, 0]
        im2 = x[:, 1]

        with tf.name_scope('im1_feed'):
            im1_convs = self.encoder(im1)

        with tf.name_scope('im2_feed'):
            im2_convs = self.encoder(im2)

        self.im1_convs = im1_convs
        self.im2_convs = im2_convs

        convs_fw = self.configure_convs(im1_convs, im2_convs, list(range(6, 0, -1)))
        self.decoder_fw_res = self.decoder(convs_fw)
        self.flow_fw = self.decoder_fw_res.pred_flow

        if self.backward_flow:
            convs_bw = self.configure_convs(im2_convs, im1_convs, list(range(6, 0, -1)))
            self.decoder_bw_res = self.decoder(convs_bw)
            self.flow_bw = self.decoder_bw_res.pred_flow

            return tf.keras.Model(inputs=x, outputs=[self.flow_fw, self.flow_bw])

        return tf.keras.Model(inputs=x, outputs=self.flow_fw)

    @staticmethod
    def supervised_loss(y_actual, y_pred):
        pred_size = tf.shape(y_pred)[1:3]
        y_actual_r = resize_flow(y_actual, pred_size, scaling=False)
        diff = y_actual_r - y_pred
        sq_diff = tf.square(diff)
        rt_diff = tf.sqrt(tf.reduce_sum(sq_diff, axis=-1, keepdims=True))
        loss = tf.reduce_mean(rt_diff, axis=[1,2,3])
        return loss


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    fake_input = np.ones([2, 2, 512, 512, 3], dtype=np.float32)

    x = tf.keras.Input(shape=(None,None,None, 3))
    pwc = PWCNet(backward_flow=True)
    model = pwc.get_model_fn(x)

    pred_flows = pwc.decoder_fw_res.mid_pred_flows
    pred_flows = pred_flows[0:-1] + [pwc.decoder_fw_res.pred_flow]

    wrapper = tf.keras.Model(inputs=x, outputs=pred_flows)


    print(model.summary())
    for _ in range(100):
        a = wrapper(fake_input)
        print("done")
        print(a[0].shape)