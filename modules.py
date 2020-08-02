import numpy as np
import tensorflow as tf
from functools import partial

import tensorflow_compression as tfc

# Feature pyramid extractor module simple/original -----------------------

class FeaturePyramidExtractor_custom_low(object):
    """ Feature pyramid extractor module"""

    def __init__(self, num_levels=6, name='fp_extractor'):
        self.num_levels = num_levels
        self.filters = [16, 24, 24, 24]
        self.name = name

    def __call__(self, images, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            features_pyramid = []
            x = images
            for l in range(self.num_levels):
                if l==0:
                    x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                else:
                    x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                features_pyramid.append(x)

            # return feature pyramid by ascent order
            return features_pyramid[::-1]

class OpticalFlowEstimator_custom_ME(object):
    """ Optical flow estimator module """
    def __init__(self, use_dc = False, name = 'of_estimator'):
        self.filters = [32, 32, 32, 32, 32]
        self.use_dc = use_dc
        self.name = name

    def __call__(self, features, flows_up_prev = None, features_up_prev = None,
                 is_output = False):
        with tf.variable_scope(self.name) as vs:
            features = features
            for f in [flows_up_prev, features_up_prev]:
                if f is not None:
                    features = tf.concat([features, f], axis = 3)

            for f in self.filters:
                conv = tf.layers.Conv2D(f, (3, 3), (1, 1), 'same')(features)
                conv = tf.nn.leaky_relu(conv, 0.1)
                if self.use_dc:
                    features = tf.concat([conv, features], axis = 3)
                else:
                    features = conv

            flows = tf.layers.Conv2D(2, (3, 3), (1, 1), 'same')(features)
            if flows_up_prev is not None:
                # Residual connection
                flows += flows_up_prev

            if is_output:
                return flows
            else:
                _, h, w, _ = tf.unstack(tf.shape(flows))
                flows_up = tf.image.resize_bilinear(flows, (2*h, 2*w))*2.0
                features_up = tf.image.resize_bilinear(features, (2*h, 2*w))
                return flows, flows_up, features_up
        

# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

class WarpingLayer(object):
    def __init__(self, warp_type = 'nearest', name = 'warping'):
        self.warp = warp_type
        self.name = name

    def __call__(self, x, flow):
        with tf.name_scope(self.name) as ns:
            assert self.warp in ['nearest', 'bilinear']
            if self.warp == 'nearest':
                x_warped = nearest_warp(x, flow)
            else:
                x_warped = bilinear_warp(x, flow)
            return x_warped
# Context module -----------------------------------------------
class FeatureExtractor_custom_RGB(object):
    def __init__(self, name='f_extractor'):
        self.filters = [6]
        self.name = name

    def __call__(self, images, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x=images
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(24, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(12, (3, 3), (1, 1), 'same')(x)
            return x

class FeatureExtractor_custom_RGB_new(object):
    """ Feature pyramid extractor module"""

    def __init__(self, name='f_extractor'):
        self.filters = [6]
        self.name = name

    def __call__(self, images, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x=images
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.1)
            return x

class FeatureExtractor_custom(object):
    """ Feature pyramid extractor module"""

    def __init__(self, name='f_extractor'):
        self.filters = [6]
        self.name = name

    def __call__(self, images, reuse=False):
        """
        Args:
        - images [Y_images, U_images, V_images]
        - Y_images (batch, h, w, 1),U_images (batch, h, w, 1),V_images (batch, h, w, 1)

        Returns:
        - features_pyramid (batch, h_l, w_l, nch_l) for each scale levels:
          extracted feature pyramid (deep -> shallow order)
        """
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            l=0
            Y=images[0]
            _, h, w, _ = tf.unstack(tf.shape(Y))
            U_up = tf.image.resize_bilinear(images[1], (h, w))
            V_up = tf.image.resize_bilinear(images[2], (h, w))
            YUV = tf.concat([Y,U_up,V_up],axis=3)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(YUV)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(24, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(12, (3, 3), (1, 1), 'same')(x)
            return x

class ContextNetwork_RGB(object):
    """ Context module """
    def __init__(self, name = 'context'):
        self.name = name

    def __call__(self, images, features):
        """
        Args:
        - flows (batch, h, w, 2): optical flow
        - features (batch, h, w, 2): feature map passed from previous OF-estimator

        Returns:
        - flows (batch, h, w, 2): convolved optical flow
        """
        with tf.variable_scope(self.name) as vs:
            x = tf.concat([images, features], axis = 3)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1),'same',
                                 dilation_rate = (2, 2))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1),'same',
                                 dilation_rate = (4, 4))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1),'same',
                                 dilation_rate = (8, 8))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(3, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            output = images + x

            return output

class ContextNetwork_RGB_ResNet(object):
    def __init__(self, name='context'):
        self.name = name

    def resblock(self, x):
        tmp = tf.nn.relu(tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x))
        tmp = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(tmp)
        return x + tmp
    def __call__(self, images, features, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(tf.concat([images,features],axis=3))
            x = self.resblock(self.resblock(self.resblock(x)))
            x = tf.nn.leaky_relu(tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x), 0.1)
            x1 = tf.nn.leaky_relu(tf.layers.Conv2D(64, (3, 3), (2, 2), 'same')(x), 0.1)
            x2 = tf.nn.leaky_relu(tf.layers.Conv2D(64, (3, 3), (2, 2), 'same')(x1), 0.1)
            _, h, w, _ = tf.unstack(tf.shape(x2))
            x2_up = tf.image.resize_bilinear(self.resblock(x2), (2*h, 2*w))
            x1 = self.resblock(x1)+x2_up
            _, h, w, _ = tf.unstack(tf.shape(x1))
            x1_up = tf.image.resize_bilinear(self.resblock(x1), (2 * h, 2 * w))
            x = self.resblock(self.resblock(x)) + x1_up
            x = self.resblock(self.resblock(self.resblock(x)))
            output = tf.layers.Conv2D(3, (3, 3), (1, 1), 'same')(x) + images
            return output

class HyperPrior_resi(object):
    def __init__(self, input_channel=128, num_filters=128, name='bls2017ImgCompression'):
        self.input_channel = input_channel
        self.num_filters = num_filters
        self.name = name

    def analysis_transform(self, tensor, num_filters):
        """Builds the analysis transform."""

        with tf.variable_scope("analysis"):
            tensor = tf.abs(tensor)
            with tf.variable_scope("layer_0"):
                tensor = tf.layers.Conv2D(num_filters, (3, 3), (1, 1), 'same')(tensor)
                tensor = tf.nn.relu(tensor)
            with tf.variable_scope("layer_1"):
                tensor = tf.layers.Conv2D(num_filters, (5, 5), (2, 2), 'same')(tensor)
                tensor = tf.nn.relu(tensor)
            with tf.variable_scope("layer_2"):
                tensor = tf.layers.Conv2D(num_filters, (5, 5), (2, 2), 'same')(tensor)
            return tensor

    def synthesis_transform(self, tensor, input_channel, num_filters):
        """Builds the synthesis transform."""

        with tf.variable_scope("synthesis"):
            with tf.variable_scope("layer_0"):
                tensor = tf.layers.Conv2DTranspose(num_filters, (5, 5), (2, 2), 'same')(tensor)
                tensor = tf.nn.relu(tensor)
            with tf.variable_scope("layer_1"):
                tensor = tf.layers.Conv2DTranspose(num_filters, (5, 5), (2, 2), 'same')(tensor)
                tensor = tf.nn.relu(tensor)
            with tf.variable_scope("layer_2"):
                tensor = tf.layers.Conv2D(input_channel, (3, 3), (1, 1), 'same')(tensor)
                tensor = tf.nn.relu(tensor)
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
            x_tilde = self.synthesis_transform(y_tilde, self.input_channel, self.num_filters) + 0.00000001
            train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
            return bit_string, entropy_bottleneck, x_tilde, train_bpp

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class FeatureExtractor_mvlf(object):
    def __init__(self, name='mvlf_extractor'):
        self.name = name
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.1)
            return x

class ContextNetwork_mv_Unet(object):
    """ Context module """
    def __init__(self, name = 'ContextNetwork_mv_Unet'):
        self.name = name

    def resblock(self, x):
        tmp = tf.nn.relu(tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x))
        tmp = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(tmp)
        return x + tmp

    def __call__(self, x, flow):
        with tf.variable_scope(self.name) as vs:
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same',
                                 dilation_rate=(1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same',
                                 dilation_rate=(2, 2))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same',
                                 dilation_rate=(4, 4))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(48, (3, 3), (1, 1), 'same',
                                 dilation_rate=(8, 8))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same',
                                 dilation_rate=(16, 16))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same',
                                 dilation_rate=(1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(2, (3, 3), (1, 1), 'same',
                                 dilation_rate=(1, 1))(x)
            return x+flow

class Resideblur_ResNet_RGB(object):
    def __init__(self, name='Resideblur_ResNet'):
        self.name = name

    def resblock(self, x):
        tmp = tf.nn.relu(tf.layers.Conv2D(48, (3, 3), (1, 1), 'same')(x))
        tmp = tf.layers.Conv2D(48, (3, 3), (1, 1), 'same')(tmp)
        return x + tmp
    def __call__(self, x, features, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x = tf.layers.Conv2D(48, (3, 3), (1, 1), 'same')(tf.concat([x,features],axis=3))
            x = self.resblock(x)
            x = tf.nn.leaky_relu(tf.layers.Conv2D(48, (3, 3), (1, 1), 'same')(x), 0.1)
            x1 = tf.nn.leaky_relu(tf.layers.Conv2D(48, (3, 3), (2, 2), 'same')(x), 0.1)
            x2 = tf.nn.leaky_relu(tf.layers.Conv2D(48, (3, 3), (2, 2), 'same')(x1), 0.1)
            _, h, w, _ = tf.unstack(tf.shape(x2))
            x2_up = tf.image.resize_bilinear(self.resblock(x2), (2*h, 2*w))
            x1 = self.resblock(x1)+x2_up
            _, h, w, _ = tf.unstack(tf.shape(x1))
            x1_up = tf.image.resize_bilinear(self.resblock(x1), (2 * h, 2 * w))
            x = self.resblock(x) + x1_up
            x = self.resblock(self.resblock(x))
            output_RGB = tf.layers.Conv2D(3, (3, 3), (1, 1), 'same')(x)
            return output_RGB