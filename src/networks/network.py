import tensorflow as tf

from lib.fast_rcnn.config import cfg
from lib.roi_pooling_layer import roi_pooling_op as roi_pool_op

from lib.rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from lib.rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from lib.rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py

# FCN pooling
from lib.psroi_pooling_layer import psroi_pooling_op as psroi_pooling_op
from lib.deform_psroi_pooling_layer import deform_psroi_pooling_op as deform_psroi_pooling_op
from lib.deform_conv_layer import deform_conv_op as deform_conv_op

from lib.networks.network import Network as OriginalNetwork

import abc
import numpy as np
from .networks_utils import colorize_tensor

DEFAULT_PADDING = 'SAME'


def include_original(dec):
    """ Meta decorator, which make the original function callable 
    (via f._original() )
    """
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator


@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)

        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)

        # Add to layer LUT.
        self.layers[name] = layer_output

        # This output is now the input for the next layer.
        self.feed(layer_output)

        # Return self for chained calls.
        return self
    return layer_decorated


class Network(OriginalNetwork):
    # Overriding some of the original code with reuse parameters--------------
    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, rate=1, biased=True,
             relu=True, padding=DEFAULT_PADDING, trainable=True,
             initializer=None, reuse=False):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]

        def convolve(i, k): return tf.nn.convolution(
            i, k, padding=padding, strides=[s_h, s_w],
            dilation_rate=[rate, rate])

        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.zeros_initializer() \
                if initializer is 'zeros' \
                else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o],
                                   init_weights, trainable,
                                   regularizer=self.l2_regularizer(
                                       cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def deform_conv(self, input, k_h, k_w, c_o, s_h, s_w, num_deform_group,
                    name, num_groups=1, rate=1, biased=True, relu=True,
                    padding=DEFAULT_PADDING, trainable=True, initializer=None,
                    reuse=False, ignore_offset=False):
        """ contribution by miraclebiu, and biased option

        Args:
            ignore_offset: True if use offset. False otherwise (equivalent to 
                common convolution) --> hacky implementation to produce vanilla
                from deformable networks
        """
        self.validate_padding(padding)
        data = input[0]
        offset = input[1]
        if ignore_offset:
            print('--> ignore offset:' + offset.op.name)
            offset *= 0
        c_i = data.get_shape()[-1]

        def trans2NCHW(x): return tf.transpose(x, [0, 3, 1, 2])

        def trans2NHWC(x): return tf.transpose(x, [0, 2, 3, 1])

        # deform conv only supports NCHW
        data = trans2NCHW(data)
        offset = trans2NCHW(offset)

        def dconvolve(i, k, o): return deform_conv_op.deform_conv_op(
            i, k, o, strides=[1, 1, s_h, s_w], rates=[1, 1, rate, rate],
            padding=padding, num_groups=num_groups,
            deformable_group=num_deform_group)

        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.zeros_initializer() \
                if initializer is 'zeros' \
                else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [c_o, c_i, k_h, k_w],
                                   init_weights, trainable,
                                   regularizer=self.l2_regularizer(
                                       cfg.TRAIN.WEIGHT_DECAY))
            print(data, kernel, offset)
            dconv = trans2NHWC(dconvolve(data, kernel, offset))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(dconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(dconv, biases)
            else:
                if relu:
                    return tf.nn.relu(dconv)
                return dconv

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride=2, name='upconv',
               biased=False, relu=True, padding=DEFAULT_PADDING,
               trainable=True, reuse=False):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            h = ((in_shape[1]) * stride)
            w = ((in_shape[2]) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name, reuse=reuse) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights = tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            filters = self.make_var(
                    'weights', filter_shape, init_weights, trainable,
                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(
                    input, filters, output_shape, strides=[1, stride, stride, 1],
                    padding=DEFAULT_PADDING, name=scope.name)

            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name,
                            reuse=False):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name, reuse=reuse) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            (rpn_labels,
             rpn_bbox_targets,
             rpn_bbox_inside_weights,
             rpn_bbox_outside_weights) = tf.py_func(
                 anchor_target_layer_py,
                 [input[0], input[1], input[2], input[3], input[4],
                  _feat_stride, anchor_scales],
                 [tf.float32, tf.float32, tf.float32, tf.float32])

            # shape is (1 x H x W x A, 2)
            rpn_labels = tf.convert_to_tensor(
                tf.cast(rpn_labels, tf.int32), name='rpn_labels')

            # shape is (1 x H x W x A, 4)
            rpn_bbox_targets = tf.convert_to_tensor(
                rpn_bbox_targets, name='rpn_bbox_targets')

            # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(
                rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')

            # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(
                rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')
            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, \
                   rpn_bbox_outside_weights

    @layer
    def proposal_target_layer(self, input, classes, name, reuse=False):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name, reuse=reuse) as scope:
            # inputs: 'rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas'
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
                = tf.py_func(proposal_target_layer_py,
                             [input[0], input[1], input[2], input[3], classes],
                             [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            # rois <- (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
            # rois = tf.convert_to_tensor(rois, name='rois')
            # goes to roi_pooling
            rois = tf.reshape(rois, [-1, 5], name='rois')
            labels = tf.convert_to_tensor(
                tf.cast(labels, tf.int32), name='labels')  # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(
                bbox_targets, name='bbox_targets')  # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(
                bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(
                bbox_outside_weights, name='bbox_outside_weights')

            self.layers['rois'] = rois

            return rois, labels, bbox_targets, bbox_inside_weights,\
                   bbox_outside_weights

    @layer
    def fc(self, input, num_out, name, relu=True, data_format="NCHW",
           trainable=True, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                if data_format == "NCHW":
                    feed_in = tf.reshape(tf.transpose(
                        input, [0, 3, 1, 2]), [-1, dim])
                elif data_format == "NHWC":
                    feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(
                    0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(
                    0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights,
                                    trainable, regularizer=self.l2_regularizer(
                                        cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def bn_scale_combo(self, input, c_in, name, relu=True, reuse=False):
        """ PVA net BN -> Scale -> Relu"""
        with tf.variable_scope(name, reuse=reuse) as scope:
            bn = self.batch_normalization._original(
                self, input, name='bn', relu=False, is_training=False)
            if relu:
                bn = tf.nn.relu(bn, name='relu')
            return bn

    @layer
    def pva_negation_block(self, input, k_h, k_w, c_o, s_h, s_w, name,
                           biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale=True, negation=True, reuse=False):
        """ for PVA net, Conv -> BN -> Neg -> Concat -> Scale -> Relu"""
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv = self.conv._original(self, input, k_h, k_w, c_o, s_h, s_w,
                                       biased=biased, relu=False, name='conv',
                                       padding=padding, trainable=trainable)
            conv = self.batch_normalization._original(
                self, conv, name='bn', relu=False, is_training=False)
            c_in = c_o
            if negation:
                conv_neg = self.negation._original(self, conv, name='neg')
                conv = tf.concat(
                    axis=3, values=[conv, conv_neg], name='concat')
                c_in += c_in
            if scale:
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in, ],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0),
                                        trainable=True,
                                        regularizer=self.l2_regularizer(0.00001))
                beta = tf.get_variable('scale/beta', shape=[c_in, ],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0),
                                       trainable=True,
                                       regularizer=self.l2_regularizer(0.00001))
                # conv = conv * alpha + beta
                conv = tf.add(tf.multiply(conv, alpha), beta)
            return tf.nn.relu(conv, name='relu')

    @layer
    def pva_negation_block_v2(self, input, k_h, k_w, c_o, s_h, s_w, c_in, name,
                              biased=True, padding=DEFAULT_PADDING,
                              trainable=True, scale=True, negation=True,
                              reuse=False):
        """ for PVA net, BN -> [Neg -> Concat ->] Scale -> Relu -> Conv"""
        with tf.variable_scope(name, reuse=reuse) as scope:
            bn = self.batch_normalization._original(
                self, input, name='bn', relu=False, is_training=False)
            if negation:
                bn_neg = self.negation._original(self, bn, name='neg')
                bn = tf.concat(axis=3, values=[bn, bn_neg], name='concat')
                c_in += c_in
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in, ],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0),
                                        trainable=True,
                                        regularizer=self.l2_regularizer(0.00004))
                beta = tf.get_variable('scale/beta', shape=[c_in, ],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0),
                                       trainable=True,
                                       regularizer=self.l2_regularizer(0.00004))
                bn = tf.add(tf.multiply(bn, alpha), beta)
            bn = tf.nn.relu(bn, name='relu')
            if name == 'conv3_1/1':
                self.layers['conv3_1/1/relu'] = bn

            conv = self.conv._original(self, bn, k_h, k_w, c_o, s_h, s_w,
                                       biased=biased, relu=False, name='conv',
                                       padding=padding, trainable=trainable)
            return conv

    @layer
    def pva_inception_res_stack(self, input, c_in, name, block_start=False,
                                type='a', reuse=False):
        if type == 'a':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 64, 24, 128, 256)
        elif type == 'b':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 96, 32, 128, 384)
        else:
            raise ('Unexpected inception-res type')
        if block_start:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(name+'/incep', reuse=reuse) as scope:
            bn = self.batch_normalization._original(
                self, input, name='bn', relu=False, is_training=False)
            bn_scale = self.scale._original(self, bn, c_in, name='bn_scale')
            # 1 x 1

            conv = self.conv._original(
                self, bn_scale, 1, 1, c_0, stride, stride, name='0/conv',
                biased=False, relu=False)
            conv_0 = self.bn_scale_combo._original(
                self, conv, c_in=c_0, name='0', relu=True)

            # 3 x 3
            bn_relu = tf.nn.relu(bn_scale, name='relu')
            if name == 'conv4_1':
                tmp_c = c_1
                c_1 = 48
            conv = self.conv._original(
                self, bn_relu, 1, 1, c_1, stride, stride, name='1_reduce/conv',
                biased=False, relu=False)
            conv = self.bn_scale_combo._original(
                self, conv, c_in=c_1, name='1_reduce', relu=True)
            if name == 'conv4_1':
                c_1 = tmp_c
            conv = self.conv._original(
                self, conv, 3, 3, c_1 * 2, 1, 1, name='1_0/conv', biased=False,
                relu=False)
            conv_1 = self.bn_scale_combo._original(
                self, conv, c_in=c_1 * 2, name='1_0', relu=True)

            # 5 x 5
            conv = self.conv._original(
                self, bn_scale, 1, 1, c_2, stride, stride, name='2_reduce/conv',
                biased=False, relu=False)
            conv = self.bn_scale_combo._original(
                self, conv, c_in=c_2, name='2_reduce', relu=True)
            conv = self.conv._original(
                self, conv, 3, 3, c_2 * 2, 1, 1, name='2_0/conv', biased=False,
                relu=False)
            conv = self.bn_scale_combo._original(
                self, conv, c_in=c_2 * 2, name='2_0', relu=True)
            conv = self.conv._original(
                self, conv, 3, 3, c_2 * 2, 1, 1, name='2_1/conv', biased=False,
                relu=False)
            conv_2 = self.bn_scale_combo._original(
                self, conv, c_in=c_2 * 2, name='2_1', relu=True)

            # pool
            if block_start:
                pool = self.max_pool._original(
                    self, bn_scale, 3, 3, 2, 2, padding=DEFAULT_PADDING,
                    name='pool')
                pool = self.conv._original(
                    self, pool, 1, 1, c_pool, 1, 1, name='poolproj/conv',
                    biased=False, relu=False)
                pool = self.bn_scale_combo._original(
                    self, pool, c_in=c_pool, name='poolproj', relu=True)

        with tf.variable_scope(name, reuse=reuse) as scope:
            if block_start:
                concat = tf.concat(
                    axis=3, values=[conv_0, conv_1, conv_2, pool], name='concat')
                proj = self.conv._original(self, input, 1, 1, c_out, 2, 2,
                                           name='proj', biased=True, relu=False)
            else:
                concat = tf.concat(
                    axis=3, values=[conv_0, conv_1, conv_2], name='concat')
                proj = input

            conv = self.conv._original(
                self, concat, 1, 1, c_out, 1, 1, name='out/conv', relu=False)
            if name == 'conv5_4':
                conv = self.bn_scale_combo._original(
                    self, conv, c_in=c_out, name='out', relu=False)
            conv = self.add._original(self, [conv, proj], name='sum')
        return conv

    @layer
    def scale(self, input, c_in, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=True,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=True,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)
    # Finish overriding--------------------------------------------------------

    # New code starts from here------------------------------------------------
    def __init__(self, n_classes, snippet_len, height, width, max_time_gap=1,
                 trainable=True):
        self.snippet_len = snippet_len
        self.n_classes = n_classes
        self.keep_prob = tf.placeholder(tf.float32)
        self.trainable = trainable
        self.max_time_gap = max_time_gap
        # assert max_time_gap >= 0 and max_time_gap < self.snippet_len, \
        #     'invalid range of max_time_gap: {}'.format(max_time_gap)

        # placeholders for rDeRF and related variables
        self.rDeRF_dict = {}
        self.offset_list_frame = []
        self.offset_list_vid = []
        self.offset_name_list = []

        # placeholders for input data and labels
        self.data_raw = tf.placeholder(
                shape=[None, self.snippet_len, height, width, 3],
                name='data_raw',
                dtype=tf.float32)
        self.labels_raw = tf.placeholder(
                shape=[None],
                name='labels_raw',
                dtype=tf.int32)

        # flatten batch of snippet into batch of frames to feed to spatial net
        # [batch_size, snippet_len, H, W, C] -> [batch_size*snippet_len, H, W, C]
        self.data = tf.reshape(
                self.data_raw,
                [-1, height, width, 3],
                name='data')

        # tiling labels for multiple frames of the same video
        # [batch_size] -> [batch_size, snippet_len]
        labels_expand = tf.expand_dims(self.labels_raw, axis=1)
        labels_tile = tf.tile(labels_expand, [1, snippet_len])
        self.labels = tf.reshape(labels_tile, [-1], name='labels')

        self.layers = dict({'data': self.data})
        self.build_graph()
        pass

    @layer
    def n_relu(self, input, name, reuse=False, epsilon=1e-5):
        """Normalization ReLU activation
        """
        with tf.variable_scope(name, reuse=reuse) as scope:
            input_relu = tf.nn.relu(input)
            max_val = tf.reduce_max(tf.abs(input_relu), axis=-1, keep_dims=True)
            output = input_relu / (max_val + epsilon)
        return output

    @layer
    def make_spacetime(self, input, name, normalization=False, reuse=False):
        """Concatenate spatial and spatial data then reshape
        """
        assert self.rDeRF_dict != {}, 'Must compute rDeRF_dict first'

        with tf.variable_scope(name, reuse=reuse) as scope:
            input_shape = input.shape.as_list()
            input_reshape = tf.reshape(input,
                                       [-1, self.snippet_len]+input_shape[1:])

            # reduce number of frames by max time gap to match rDerF
            spacetime_feat = [input_reshape[:, :-self.max_time_gap, ...]]

            # combine spatial and temporal feature
            for gap in range(1, self.max_time_gap+1):
                spacetime_feat += self.rDeRF_dict['gap'+str(gap)]
            spacetime_feat = tf.concat(spacetime_feat, axis=-1)
        return spacetime_feat

    @layer
    def make_space(self, input, name, reuse=False):
        """Reshape spatial feature
        """
        with tf.variable_scope(name, reuse=reuse) as scope:
            input_shape = input.shape.as_list()
            input_reshape = tf.reshape(input,
                                       [-1, self.snippet_len]+input_shape[1:])
            input_reshape = [input_reshape[:, :-self.max_time_gap, ...]]
        return input_reshape

    @layer
    def max_pool3d(self, input, k_d, k_h, k_w, s_d, s_h, s_w, name,
                   padding=DEFAULT_PADDING):
        """Max pooling for time domain
        """
        assert len(input.shape) == 5
        return tf.nn.max_pool3d(input,
                                ksize=[1, k_d, k_h, k_w, 1],
                                strides=[1, s_d, s_h, s_w, 1],
                                padding=padding,
                                name=name)

    @layer
    def reduce_time(self, input, name, reuse=False):
        """Reduce time dimension by averaging
        """
        if isinstance(input, list):
            assert len(input) == 1
            input = input[0]
        assert len(input.shape) == 5
        with tf.variable_scope(name, reuse=reuse) as scope:
            output = tf.reduce_mean(input, axis=1)
        return output

    @layer
    def conv3d_preserve(self, input, k_d, k_h, k_w, c_o, s_d, s_h, s_w, name,
                        reduce_time=False, rate=1, biased=True, relu=True,
                        padding='VALID', trainable=True, initializer=None,
                        reuse=False):
        """3D convolution layer wrapper with spatial dimension preserving

        Args:
            k_d, k_h, k_w: kernel sizes for depth, height, width
            c_o: number of output channels
            s_d, s_h, s_w: strides for depth, heightm, width
            reduce_time: whether to reduce time dimension
        """
        # TODO: make this flexible for other types of strides and kernel sizes
        assert s_h == 1 and s_w == 1
        assert k_h == 3 and k_w == 3

        # self.validate_padding(padding)
        c_i = input.get_shape()[-1]

        # pad input on spatial dimension to retain spatial dimension
        if padding == 'VALID':
            paddings = tf.constant([[0, 0],  # batch
                                    [0, 0],  # time
                                    [1, 1],  # height
                                    [1, 1],  # width
                                    [0, 0]])  # channels
            input = tf.pad(input, paddings, 'CONSTANT')

        def convolve(input, kernel):
            output = tf.nn.convolution(input, kernel, padding=padding,
                                       strides=[s_d, s_h, s_w],
                                       dilation_rate=[rate, rate, rate],
                                       data_format='NDHWC')
            if reduce_time:
                output = tf.reduce_mean(output, axis=1)
            return output

        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.zeros_initializer() \
                if initializer is 'zeros' \
                else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            # init_biases = tf.constant_initializer(0.0)
            init_biases = tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            kernel = self.make_var('weights', [k_d, k_h, k_w, c_i, c_o],
                                   init_weights, trainable,
                                   regularizer=self.l2_regularizer(
                                       cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv
        pass  # end of conv3d layer

    @abc.abstractmethod
    def build_graph(self):
        """Build graph for model
        """
        pass

    @abc.abstractmethod
    def retrieve_offsets(self):
        """Retrieve deformable offsets defined in the network. Return list of
        offsets corresponding to a network
        """
        offset_list = []
        offset_name_list = []
        return offset_list, offset_name_list

    def reshape_offsets(self, offset_list, snippet_len):
        """Reshape a list of offet so that batch of frames becomes batch of
        snippet to get motion loss for each snippet.

        Args:
            offset_list: list of offsets, each item is a tensor
            snippet_len: len of snippet to reshape into

        Returns:
            output: reshaped offsets
        """
        output = []
        for offset in offset_list:
            offset_shape = offset.shape.as_list()
            tmp = tf.reshape(offset, [-1, snippet_len] + offset_shape[1:])
            output.append(tmp)
        return output

    def compute_class_loss(self):
        """Compute cross entropy loss as class loss

        Return:
            Cross entropy class loss
        """
        cls_score = self.get_output('cls_score')
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cls_score, labels=self.labels_raw)
        cross_entropy = tf.reduce_mean(cross_entropy_n)
        return tf.identity(cross_entropy, name='class_loss')

    def compute_rDeRF(self, normalization=None, space_name='', g_var=None):
        """Retrieve the list of rDeRF (temporal features)

        Args:
            normalization: type of normalization.
                None: do not normalize temporal features
                `dynamic`: normalize by the distribution of spatial features
                `global`: normalize by given g_mean ad g_var
            space_name: name of layer providing spatial features. Must be
                given if `normalization` is not None. The shape of its output
                is (?, H, W, C). It will be reshaped as (?, S, H, W, C) where S
                is snippet_len
            g_var: global var. Must be given if `normalization` is `global`

        Returns:
            rDeRF_dict: dict of rDeRF pack, each item is an rDeRF list at a time
                gap, each item of rDeRF list is an rDeRF at a resolution. E.g.
                    rDeRF_dict['gap1'] = [rDeRF@5a, rDeRF@5b, rDeRF@5c],
                    rDeRF_dict['gap2'] = [rDeRF@5a, rDeRF@5b, rDeRF@5c],
                    etc.
            offset_list_frame: list of deformable offsets in frame level
            offset_list_vid: list of deformable offsets in video level,
                obtained by reshaping offset_list_frame with snippet_len
            offset_name_list: name of the offsets
        """
        if normalization is not None:
            assert space_name != '', 'Must provide name for spatial features'
            if normalization == 'dynamic':
                space_feat = self.get_output(space_name)
                space_shape = space_feat.shape.as_list()
                space_feat = tf.reshape(space_feat, [-1, self.snippet_len]+space_shape[1:])
                space_feat = space_feat[:, :-self.max_time_gap, ...]
                _, s_var = tf.nn.moments(space_feat, axes=[1, 2, 3, 4])
            elif normalization == 'global':
                assert g_var is not None, 'Must provide global variance'
                s_var = g_var
            else:
                print('Unsupported normalization type')
                exit(-1)

        # only compute if not done it before to avoid duplicating variables
        if self.rDeRF_dict == {}:
            max_time_gap = self.max_time_gap
            offset_list_frame, offset_name_list = self.retrieve_offsets()
            offset_list_vid = self.reshape_offsets(offset_list_frame, self.snippet_len)

            rDeRF_dict = {}
            for gap in range(1, max_time_gap+1):
                rDeRF_curent_gap = []
                i = 0
                for offset in offset_list_vid:
                    name = 'rDeRF_gap{}_{}'.format(gap, offset_name_list[i])
                    rDeRF = tf.subtract(offset[:, max_time_gap:],
                                        offset[:, max_time_gap-gap:-gap],
                                        name=name)

                    # Normalization
                    if normalization is not None:
                        # temporal mean and variance, shape=(?,)
                        t_mean, t_var = tf.nn.moments(rDeRF, axes=[1, 2, 3, 4])

                        # (?,15,14,14,72) -> (15,14,14,72,?) for broadcasting
                        rDeRF = tf.transpose(rDeRF, [1, 2, 3, 4, 0])

                        # normalization
                        rDeRF = (rDeRF-t_mean) * tf.sqrt(s_var/t_var) + t_mean

                        # transpose back
                        rDeRF = tf.transpose(rDeRF, [4, 0, 1, 2, 3])

                    rDeRF_curent_gap.append(rDeRF)
                    i += 1
                rDeRF_dict['gap'+str(gap)] = rDeRF_curent_gap

            # store in network
            self.rDeRF_dict = rDeRF_dict
            self.offset_list_frame = offset_list_frame
            self.offset_list_vid = offset_list_vid
            self.offset_name_list = offset_name_list
        pass

    def compute_motion_loss(self, scaling_factor, gap_scales):
        """ compute motion loss from offsets list (at multple resolutions)

        Args:
            scaling_factor: float number indicating the scaling factor to
                multiply with motion loss (default = 1.0 <=> no scaling)
            max_time_gap: maximum of time widening to apply. Must be at least
                1 and less than snippet_len
            gap_scales: scale for each temporal gap, ndarray with length of
                max_time_gap

        Returns:
            Motion loss
        """
        rDeRF_dict = self.rDeRF_dict
        motion_loss = 0.0

        # for each temporal gap
        for gap in range(1, self.max_time_gap+1):
            # retrieve rDeRF list of all resolutions
            rDeRF_list = rDeRF_dict['gap'+str(gap)]

            # compute the motion loss of the current gap
            motion_loss_current_gap = 0.0
            for rDeRF in rDeRF_list:
                motion_loss_current_gap += tf.reduce_mean(rDeRF ** 2)

            # add to the general motion loss
            motion_loss += gap_scales[gap-1] * motion_loss_current_gap

        # rescale the general motion loss
        motion_loss *= scaling_factor
        return tf.identity(motion_loss, name='motion_loss')

    def compute_total_loss(self, usemotionloss=True, scaling_factor=1.0,
                           gap_scales=None):
        """ Compute total loss.

        Args:
            usemotionloss: whether to include motion_loss in total_loss
            scaling_factor: float number indicating the scaling factor to
                multiply with motion loss (default = 1.0 <=> no scaling)

        Returns:
            total_loss = regularization_loss + class_loss (+ motion_loss)
            class_loss: class loss
            motion_loss: motion loss, or None if usemotionloss is False
        """
        # progress gap_scales to create a ndarray of float32
        if gap_scales is None:
            gap_scales = np.ones(self.max_time_gap, dtype=np.float32)
        elif isinstance(gap_scales, str):
            gap_scales = gap_scales.split(',')
            gap_scales = [float(x) for x in gap_scales]
            gap_scales = np.array(gap_scales, dtype=np.float32)
        elif isinstance(gap_scales, list):
            gap_scales = np.array(gap_scales, dtype=np.float32)

        # compute losses
        regularization_losses = tf.add_n(tf.losses.get_regularization_losses(),
                                         name='regularization_loss')
        class_loss = self.compute_class_loss()
        total_loss = 1e-4*regularization_losses + class_loss

        # include motion loss on demand
        if usemotionloss:
            motion_loss = self.compute_motion_loss(scaling_factor, gap_scales)
            total_loss += motion_loss
        else:
            motion_loss = None

        loss_dict = {'class_loss': class_loss,
                     'motion_loss': motion_loss}
        return tf.identity(total_loss, name='total_loss'), loss_dict

    def infer(self):
        """Run inference on the network

        Return:
            class probability of the network
        """
        return self.get_output('cls_prob')

    def create_metrics_old(self):
        """ Create validation metrics on frame and video levels

        Returns:
            accuracy_frame: accuracy on frame level
            accuracy_vid: accuracy on video level
            confusion_frame: unnormalized confusion matrix on frame level
            confusion_vid: unnormalized confusion matrix on video level
            metrics_op: group of metrics updating operations
        """
        inference = self.infer()

        # accuracy metrics-----------------------------------------------------
        # frame-level accuracy
        predictions_frame = tf.argmax(inference, axis=1)
        accuracy_frame, accuracy_frame_update = tf.metrics.accuracy(
                labels=self.labels, predictions=predictions_frame,
                name='accuracy_frame')

        # video-level accuracy
        inference_reshape = tf.reshape(
                inference, [-1, self.snippet_len, self.n_classes])
        # compute video probability by averaging across the frames
        video_prob = tf.reduce_mean(inference_reshape, axis=1)
        predictions_vid = tf.argmax(video_prob, axis=1)
        accuracy_vid, accuracy_vid_update = tf.metrics.accuracy(
                labels=self.labels_raw, predictions=predictions_vid,
                name='accuracy_vid')

        # confusion matrix-----------------------------------------------------
        # frame-level
        batch_confusion_frame = tf.confusion_matrix(
                labels=self.labels, predictions=predictions_frame,
                num_classes=self.n_classes)
        confusion_frame = tf.Variable(
                tf.zeros([self.n_classes, self.n_classes], dtype=tf.int32),
                name='confusion_frame')
        confusion_frame_update = confusion_frame.assign(
                confusion_frame + batch_confusion_frame)
        # confusion_frame_img = colorize_tensor(confusion_frame, cmap='hot',
                                              # extend=True)

        # video-level
        batch_confusion_vid = tf.confusion_matrix(
                labels=self.labels_raw, predictions=predictions_vid,
                num_classes=self.n_classes)
        confusion_vid = tf.Variable(
                tf.zeros([self.n_classes, self.n_classes], dtype=tf.int32),
                name='confusion_vid')
        confusion_vid_update = confusion_vid.assign(
                confusion_vid + batch_confusion_vid)
        # confusion_vid_img = colorize_tensor(confusion_vid, cmap='hot',
                                            # extend=True)

        # grouping metrics
        metrics_op = tf.group(accuracy_frame_update,
                              accuracy_vid_update,
                              confusion_frame_update,
                              confusion_vid_update)
        return accuracy_frame, accuracy_vid, confusion_frame, confusion_vid, \
            metrics_op

    def create_metrics(self):
        """ Create validation metrics on frame and video levels

        Returns:
            accuracy_vid: accuracy on video level
            confusion_vid: unnormalized confusion matrix on video level
            metrics_op: group of metrics updating operations
        """
        inference = self.infer()

        # accuracy metrics-----------------------------------------------------
        # compute video probability by averaging across the frames
        predictions_vid = tf.argmax(inference, axis=1)
        accuracy_vid, accuracy_vid_update = tf.metrics.accuracy(
                labels=self.labels_raw, predictions=predictions_vid,
                name='accuracy_vid')

        # confusion matrix-----------------------------------------------------
        batch_confusion_vid = tf.confusion_matrix(
                labels=self.labels_raw, predictions=predictions_vid,
                num_classes=self.n_classes)
        confusion_vid = tf.Variable(
                tf.zeros([self.n_classes, self.n_classes], dtype=tf.int32),
                name='confusion_vid')
        confusion_vid_update = confusion_vid.assign(
                confusion_vid + batch_confusion_vid)

        # grouping metrics
        metrics_op = tf.group(accuracy_vid_update,
                              confusion_vid_update)
        return accuracy_vid, confusion_vid, metrics_op
