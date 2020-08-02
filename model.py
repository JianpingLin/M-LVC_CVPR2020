from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import *

import tensorflow_compression as tfc
import numpy as np

class MAMVPNet(object):
    def __init__(self, num_levels=6,
                 warp_type='bilinear', use_dc=False,
                 output_level=4, name='pwcmenet'):
        self.num_levels = num_levels
        self.warp_type = warp_type
        self.use_dc = use_dc
        self.output_level = output_level
        self.name = name

        self.fp_extractor = FeaturePyramidExtractor_custom_low(self.num_levels)
        self.warp_layer = WarpingLayer(self.warp_type)
        self.of_estimators = [OpticalFlowEstimator_custom_ME(use_dc=self.use_dc, name=f'optflow_{l}') \
                              for l in range(self.num_levels + 1)]

    def __call__(self, flows_2_pyramid, flows_1_pyramid, flows_0_pyramid, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            pyramid_0 = self.fp_extractor(flows_0_pyramid[-1], reuse=reuse)
            pyramid_1 = self.fp_extractor(flows_1_pyramid[-1])
            pyramid_2 = self.fp_extractor(flows_2_pyramid[-1])

            pyramid_1_warped = []
            pyramid_2_warped = []
            for l, (flows_0, flows_1, features_1, features_2) in enumerate(
                    zip(flows_0_pyramid, flows_1_pyramid, pyramid_1, pyramid_2)):
                print(f'Warp Optical Flow Level {l}')

                features_1_warped = self.warp_layer(features_1, flows_0)
                pyramid_1_warped.append(features_1_warped)
                features_2_warped = self.warp_layer(features_2, (flows_0 + self.warp_layer(flows_1, flows_0)))
                pyramid_2_warped.append(features_2_warped)

            flows_pyramid = []
            flows_up, features_up = None, None
            for l, (features_0, features_1, features_2) in enumerate(
                    zip(pyramid_0, pyramid_1_warped, pyramid_2_warped)):
                print(f'Level {l}')

                # Optical flow estimation
                features_total = tf.concat([features_0, features_1, features_2], axis=3)
                if l < self.output_level:
                    flows, flows_up, features_up \
                        = self.of_estimators[l](features_total, flows_up, features_up)
                else:
                    # At output level
                    flows = self.of_estimators[l](features_total, flows_up, features_up,
                                                  is_output=True)
                    return flows, None, None

                flows_pyramid.append(flows)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class MCNet_Multiple(object):
    def __init__(self, name='mcnet'):
        self.name = name
        self.warp_layer = WarpingLayer('bilinear')
        self.f_extractor = FeatureExtractor_custom_RGB_new()
        self.context_RGB = ContextNetwork_RGB_ResNet(name='context_RGB')

    def __call__(self, images_pre_rec_4, images_pre_rec_3, images_pre_rec_2, images_pre_rec, flow3, flow2, flow1, flow, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            images_pre_rec_warped = self.warp_layer(images_pre_rec, flow*20.0)
            features = self.f_extractor(images_pre_rec)
            features_warped = self.warp_layer(features, flow*20.0)
            flow1_warped = self.warp_layer(flow1, flow * 20.0)
            flow2_warped = self.warp_layer(flow2, (flow1_warped + flow) * 20.0)
            flow3_warped = self.warp_layer(flow3, (flow2_warped + flow1_warped + flow) * 20.0)
            features_4 = self.f_extractor(images_pre_rec_4, reuse=True)
            features_4_warped = self.warp_layer(features_4, (flow3_warped + flow2_warped + flow1_warped + flow) * 20.0)
            features_3 = self.f_extractor(images_pre_rec_3, reuse=True)
            features_3_warped = self.warp_layer(features_3, (flow2_warped + flow1_warped + flow) * 20.0)
            features_2 = self.f_extractor(images_pre_rec_2, reuse=True)
            features_2_warped = self.warp_layer(features_2, (flow1_warped + flow) * 20.0)
            features = tf.concat([features_4_warped, features_3_warped, features_2_warped, features_warped], axis=3)
            output = self.context_RGB(images_pre_rec_warped, features)
            return output, features

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    @property
    def vars_restore(self):
        return [var for var in tf.global_variables() if ((self.name in var.name) and ('context_RGB' not in var.name))]

class MCNet(object):
    def __init__(self, name='mcnet'):
        self.name = name
        self.warp_layer = WarpingLayer('bilinear')
        self.f_extractor = FeatureExtractor_custom_RGB()
        self.context_RGB = ContextNetwork_RGB_ResNet(name='context_RGB')
        # self.scales = [None, 0.625, 1.25, 2.5, 5.0, 10., 20.]

    def __call__(self, images_pre_rec, flow, reuse=False):
        """Y_frames (n, h, w, 1)"""
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            images_pre_rec_warped = self.warp_layer(images_pre_rec, flow)
            features = self.f_extractor(images_pre_rec)
            features_warped = self.warp_layer(features, flow)
            output = self.context_RGB(images_pre_rec_warped, features_warped)
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    @property
    def vars_restore(self):
        return [var for var in tf.global_variables() if
                ((self.name in var.name) and ('context_RGB' not in var.name) and ('f_extractor' not in var.name))]

class ResiDeBlurNet(object):
    def __init__(self, name='resideblurmodel'):
        self.name = name
        self.f_extractor = FeatureExtractor_custom_RGB_new()
        self.warp_layer = WarpingLayer('bilinear')
        self.resideblur_ResNet=Resideblur_ResNet_RGB('Resideblur_ResNet')
        # self.scales = [None, 0.625, 1.25, 2.5, 5.0, 10., 20.]

    def __call__(self, tensor, images_pred, features_warped, reuse=False):
        """Y_frames (n, h, w, 1)"""
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            features = self.f_extractor(images_pred)
            features = tf.concat([features_warped, features], axis=3)
            output = self.resideblur_ResNet(tensor, features)
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class bls2017ImgCompression_mvd_factor(object):
    def __init__(self, input_channel=2, num_filters=128, name='bls2017ImgCompression'):
        self.input_channel=input_channel
        self.num_filters=num_filters
        self.name = name

    def analysis_transform(self, tensor, num_filters):
        """Builds the analysis transform."""

        with tf.variable_scope("analysis"):
            with tf.variable_scope("layer_0"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN())
                tensor = layer(tensor)
            with tf.variable_scope("layer_1"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN())
                tensor = layer(tensor)
            with tf.variable_scope("layer_2"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN())
                tensor = layer(tensor)

            with tf.variable_scope("layer_3"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=False, activation=None)
                tensor = layer(tensor)

            return tensor

    def synthesis_transform(self, tensor, input_channel,  num_filters):
        """Builds the synthesis transform."""

        with tf.variable_scope("synthesis"):
            with tf.variable_scope("layer_0"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN(inverse=True))
                tensor = layer(tensor)
            with tf.variable_scope("layer_1"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN(inverse=True))
                tensor = layer(tensor)
            with tf.variable_scope("layer_2"):
                layer = tfc.SignalConv2D(
                    num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN(inverse=True))
                tensor = layer(tensor)
            with tf.variable_scope("layer_3"):
                layer = tfc.SignalConv2D(
                    input_channel, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=None)
                tensor = layer(tensor)

            return tensor
    def __call__(self, x, num_pixels, reuse=False, isTrain=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            y = self.analysis_transform(x, self.num_filters)
            entropy_bottleneck = tfc.EntropyBottleneck()
            bit_string = None
            if isTrain:
                y_tilde, likelihoods = entropy_bottleneck(y, training=True)
            else:
                string = entropy_bottleneck.compress(y)
                bit_string = tf.squeeze(string, axis=0)
                y_tilde, likelihoods = entropy_bottleneck(y, training=False)
            x_tilde = self.synthesis_transform(y_tilde, self.input_channel, self.num_filters)
            train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
            return bit_string, entropy_bottleneck, x_tilde, train_bpp

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class bls2017ImgCompression_resi_RGB(object):
    def __init__(self, input_channel=2, N_filters=128,M_filters=128, name='bls2017ImgCompression'):
        self.input_channel = input_channel
        self.N_filters = N_filters
        self.M_filters = M_filters
        self.name = name
        self.hyperModel=HyperPrior_resi(M_filters,N_filters,'hyper_resi')

    def analysis_transform(self, tensor, N_filters, M_filters):
        """Builds the analysis transform."""

        with tf.variable_scope("analysis"):
            with tf.variable_scope("layer_0"):
                layer = tfc.SignalConv2D(
                    N_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN())
                tensor = layer(tensor)
            with tf.variable_scope("layer_1"):
                layer = tfc.SignalConv2D(
                    N_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN())
                tensor = layer(tensor)
            with tf.variable_scope("layer_2"):
                layer = tfc.SignalConv2D(
                    N_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN())
                tensor = layer(tensor)

            with tf.variable_scope("layer_3"):
                layer = tfc.SignalConv2D(
                    M_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                    use_bias=False, activation=None)
                tensor = layer(tensor)

            return tensor

    def synthesis_transform(self, tensor, N_filters):
        """Builds the synthesis transform."""

        with tf.variable_scope("synthesis"):
            with tf.variable_scope("layer_0"):
                layer = tfc.SignalConv2D(
                    N_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN(inverse=True))
                tensor = layer(tensor)
            with tf.variable_scope("layer_1"):
                layer = tfc.SignalConv2D(
                    N_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN(inverse=True))
                tensor = layer(tensor)
            with tf.variable_scope("layer_2"):
                layer = tfc.SignalConv2D(
                    N_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=tfc.GDN(inverse=True))
                tensor = layer(tensor)
            with tf.variable_scope("layer_out"):
                layer = tfc.SignalConv2D(
                    32, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                    use_bias=True, activation=None)
                tensor = layer(tensor)
            '''
            with tf.variable_scope("layer_4"):
                layer = tfc.SignalConv2D(
                    3, (3, 3), corr=False, strides_up=1, padding="same_zeros",
                    use_bias=True, activation=None)
                tensor = layer(tensor)
            '''

            return tensor

    def __call__(self, resi_frames, num_pixels, reuse=False, isTrain=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:

            y = self.analysis_transform(resi_frames, self.N_filters, self.M_filters)

            entropy_bottleneck = tfc.EntropyBottleneck_gauss()
            bit_string = None
            bit_string_dev = None
            if isTrain:
                _, entropy_bottleneck_dev, dev_tilde, train_bpp_dev = self.hyperModel(y, num_pixels, reuse=False, isTrain=True)
                y_tilde, likelihoods = entropy_bottleneck(y, dev_tilde, training=True)
            else:
                bit_string_dev, entropy_bottleneck_dev, dev_tilde, train_bpp_dev = self.hyperModel(y, num_pixels,
                                                                                               reuse=False,
                                                                                               isTrain=False)
                with tf.device("/cpu:0"):
                    string = entropy_bottleneck.compress(y, dev_tilde)
                bit_string = tf.squeeze(string, axis=0)
                y_tilde, likelihoods = entropy_bottleneck(y, dev_tilde, training=False)
            tensor_tilde = self.synthesis_transform(y_tilde, self.N_filters)
            train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
            return bit_string, entropy_bottleneck, tensor_tilde, train_bpp, bit_string_dev, entropy_bottleneck_dev, dev_tilde, train_bpp_dev,

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class MVLoopFiltering(object):
    def __init__(self, name='mvlfmodel'):
        self.name = name
        self.f_extractor_0=FeatureExtractor_mvlf('FeatureExtractor_ilf_0')
        self.f_extractor_1 = FeatureExtractor_mvlf('FeatureExtractor_ilf_1')
        self.f_extractor_2 = FeatureExtractor_mvlf('FeatureExtractor_ilf_2')
        self.warp_layer = WarpingLayer('bilinear')
        self.context_mv_Unet = ContextNetwork_mv_Unet(name='context_mv_Unet')
        # self.scales = [None, 0.625, 1.25, 2.5, 5.0, 10., 20.]

    def __call__(self, flow3, flow2, flow1, flow, images_pre_rec, reuse=False):
        """Y_frames (n, h, w, 1)"""
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            flow3_f = self.f_extractor_0(flow3)
            flow2_f = self.f_extractor_0(flow2, reuse=True)
            flow1_f = self.f_extractor_0(flow1, reuse=True)
            flow_f = self.f_extractor_1(flow)
            images_pre_rec_f = self.f_extractor_2(images_pre_rec)
            flow1_warped = self.warp_layer(flow1, flow * 20.0)
            flow2_warped = self.warp_layer(flow2, (flow1_warped + flow) * 20.0)

            flow3_f_warped = self.warp_layer(flow3_f, (flow2_warped + flow1_warped + flow) * 20.)
            flow2_f_warped = self.warp_layer(flow2_f, (flow1_warped + flow) * 20.)
            flow1_f_warped = self.warp_layer(flow1_f, flow * 20.)
            input = tf.concat([flow3_f_warped, flow2_f_warped, flow1_f_warped, flow_f, images_pre_rec_f], axis=3)
            output = self.context_mv_Unet(input, flow)
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]