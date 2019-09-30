from .network import Network, layer
import tensorflow as tf
from data_utils import misc_utils


class Resnet50VanillaOpticalFlow(Network):
    def __init__(self, n_classes, snippet_len, height, width, max_time_gap,
                 trainable):
        # Define groundtruth optical flow for training
        self.use_flow = True
        self.flow_gt = tf.placeholder(
            shape=[None, snippet_len-max_time_gap, height, width, 2],
            name='flow_gt', dtype=tf.float32)
        self.flow = tf.reshape(
            self.flow_gt, [-1, height, width, 2], name='flow')

        # call supper init
        super().__init__(n_classes, snippet_len, height, width, max_time_gap,
                         trainable)
        pass

    def build_graph(self):
        """ Build graph for model
        """
        self.layers['flow'] = self.flow
        n_classes = self.n_classes

        # ---------------------------------------------------------------------
        # Appearance stream
        # Feature extraction
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
             .batch_normalization(relu=True, name='bn_conv1', is_training=False)
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(name='bn2a_branch1', is_training=False, relu=False))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(relu=True, name='bn2a_branch2a', is_training=False)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(relu=True, name='bn2a_branch2b', is_training=False)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(name='bn2a_branch2c', is_training=False, relu=False))

        (self.feed('bn2a_branch1', 'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(relu=True, name='bn2b_branch2a', is_training=False)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(relu=True, name='bn2b_branch2b', is_training=False)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(name='bn2b_branch2c', is_training=False, relu=False))

        (self.feed('res2a_relu', 'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(relu=True, name='bn2c_branch2a', is_training=False)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(relu=True, name='bn2c_branch2b', is_training=False)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(name='bn2c_branch2c', is_training=False, relu=False))

        (self.feed('res2b_relu', 'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1',
                   padding='VALID')
             .batch_normalization(name='bn3a_branch1', is_training=False, relu=False))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a',
                   padding='VALID')
             .batch_normalization(relu=True, name='bn3a_branch2a', is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(relu=True, name='bn3a_branch2b', is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(name='bn3a_branch2c', is_training=False, relu=False))

        (self.feed('bn3a_branch1', 'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
             .batch_normalization(relu=True, name='bn3b_branch2a', is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
             .batch_normalization(relu=True, name='bn3b_branch2b', is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
             .batch_normalization(name='bn3b_branch2c', is_training=False, relu=False))

        (self.feed('res3a_relu', 'bn3b_branch2c')
             .add(name='res3b')
             .relu(name='res3b_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
             .batch_normalization(relu=True, name='bn3c_branch2a', is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
             .batch_normalization(relu=True, name='bn3c_branch2b', is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
             .batch_normalization(name='bn3c_branch2c', is_training=False, relu=False))

        (self.feed('res3b_relu', 'bn3c_branch2c')
             .add(name='res3c')
             .relu(name='res3c_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
             .batch_normalization(relu=True, name='bn3d_branch2a', is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
             .batch_normalization(relu=True, name='bn3d_branch2b', is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
             .batch_normalization(name='bn3d_branch2c', is_training=False, relu=False))

        (self.feed('res3c_relu', 'bn3d_branch2c')
             .add(name='res3d')
             .relu(name='res3d_relu')
             .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1',
                   padding='VALID')
             .batch_normalization(name='bn4a_branch1', is_training=False, relu=False))

        (self.feed('res3d_relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a',
                   padding='VALID')
             .batch_normalization(relu=True, name='bn4a_branch2a', is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(relu=True, name='bn4a_branch2b', is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(name='bn4a_branch2c', is_training=False, relu=False))

        (self.feed('bn4a_branch1', 'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
             .batch_normalization(relu=True, name='bn4b_branch2a', is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
             .batch_normalization(relu=True, name='bn4b_branch2b', is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
             .batch_normalization(name='bn4b_branch2c', is_training=False, relu=False))

        (self.feed('res4a_relu', 'bn4b_branch2c')
             .add(name='res4b')
             .relu(name='res4b_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
             .batch_normalization(relu=True, name='bn4c_branch2a', is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
             .batch_normalization(relu=True, name='bn4c_branch2b', is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
             .batch_normalization(name='bn4c_branch2c', is_training=False, relu=False))

        (self.feed('res4b_relu', 'bn4c_branch2c')
             .add(name='res4c')
             .relu(name='res4c_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
             .batch_normalization(relu=True, name='bn4d_branch2a', is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
             .batch_normalization(relu=True, name='bn4d_branch2b', is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
             .batch_normalization(name='bn4d_branch2c', is_training=False, relu=False))

        (self.feed('res4c_relu', 'bn4d_branch2c')
             .add(name='res4d')
             .relu(name='res4d_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
             .batch_normalization(relu=True, name='bn4e_branch2a', is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
             .batch_normalization(relu=True, name='bn4e_branch2b', is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
             .batch_normalization(name='bn4e_branch2c', is_training=False, relu=False))

        (self.feed('res4d_relu', 'bn4e_branch2c')
             .add(name='res4e')
             .relu(name='res4e_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
             .batch_normalization(relu=True, name='bn4f_branch2a', is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
             .batch_normalization(relu=True, name='bn4f_branch2b', is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
             .batch_normalization(name='bn4f_branch2c', is_training=False, relu=False))

        (self.feed('res4e_relu', 'bn4f_branch2c')
             .add(name='res4f')
             .relu(name='res4f_relu'))

        '''
        #========= RPN ============
        (self.feed('res4f_relu')
             .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*2, 1, 1, padding='VALID',
                   relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))
        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*4, 1, 1, padding='VALID',
                   relu=False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .spatial_reshape_layer(len(anchor_scales)*3*2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes', 'gt_ishard', 'dontcare_areas')
             .proposal_target_layer(n_classes, name='roi-data'))
        '''

        # Deformable
        (self.feed('res4f_relu')
            .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5a_branch1'))

        (self.feed('res4f_relu')
            .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5a_branch2a')
            .relu(name='res5a_branch2a_relu')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
                  name='res5a_branch2b_offset', padding='SAME', initializer='zeros'))

        (self.feed('res5a_branch2a_relu', 'res5a_branch2b_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=False,
                         num_deform_group=4, name='res5a_branch2b',
                         ignore_offset=True)  # deform but not actually
            .batch_normalization(relu=False, name='bn5a_branch2b')
            .relu(name='res5a_branch2b_relu')
            .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1', 'bn5a_branch2c')
            .add(name='res5a')
            .relu(name='res5a_relu')
            .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5b_branch2a')
            .relu(name='res5b_branch2a_relu')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
                  name='res5b_branch2b_offset', padding='SAME', initializer='zeros'))

        (self.feed('res5b_branch2a_relu', 'res5b_branch2b_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=False,
                         num_deform_group=4, name='res5b_branch2b',
                         ignore_offset=True)  # deform but not actually
            .batch_normalization(relu=False, name='bn5b_branch2b')
            .relu(name='res5b_branch2b_relu')
            .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5b_branch2c'))

        (self.feed('res5a_relu', 'bn5b_branch2c')
            .add(name='res5b')
            .relu(name='res5b_relu')
            .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5c_branch2a')
            .relu(name='res5c_branch2a_relu')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
                  name='res5c_branch2b_offset', padding='SAME', initializer='zeros'))

        (self.feed('res5c_branch2a_relu', 'res5c_branch2b_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=False,
                         num_deform_group=4, name='res5c_branch2b',
                         ignore_offset=True)  # deform but not actually
            .batch_normalization(relu=False, name='bn5c_branch2b')
            .relu(name='res5c_branch2b_relu')
            .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5c_branch2c'))

        (self.feed('res5b_relu', 'bn5c_branch2c')
            .add(name='res5c')
            .relu(name='res5c_relu')
            .conv(1, 1, 256, 1, 1, relu=False, name='conv_new_1')
            .relu(name='conv_new_1_relu'))

        '''
        (self.feed('conv_new_1_relu', 'roi-data')
            .deform_psroi_pool(group_size=1, pooled_size=7, sample_per_part=4,
                               no_trans=True, part_size=7, output_dim=256,
                               trans_std=1e-1, spatial_scale=0.0625, name='offset_t')
            # .flatten_data(name='offset_flatten')
            .fc(num_out=7 * 7 * 2, name='offset', relu=False)
            .reshape(shape=(-1, 2, 7, 7), name='offset_reshape'))

        (self.feed('conv_new_1_relu', 'roi-data', 'offset_reshape')
            .deform_psroi_pool(group_size=1, pooled_size=7, sample_per_part=4,
                               no_trans=False, part_size=7, output_dim=256,
                               trans_std=1e-1, spatial_scale=0.0625,
                               name='deformable_roi_pool')
            .fc(num_out=1024, name='fc_new_1')
            .fc(num_out=1024, name='fc_new_2'))
        '''

        # ---------------------------------------------------------------------
        # compute rDeRF
        self.compute_rDeRF()  # dummy

        # Combining spatial and temporal
        (self.feed('conv_new_1_relu')
            .make_space(name='space_feat')
            .reduce_time(name='space_reduce'))

        # Classification
        (self.feed('space_reduce')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool_new')
            .reshape(shape=(-1, 7, 7, 256), name='pool_new_reshape')
            .fc(num_out=1024, name='fc_new_1')
            .fc(num_out=1024, name='fc_new_2'))

        (self.feed('fc_new_2')
            .fc(num_out=n_classes, name='space_score', relu=False))

        # ---------------------------------------------------------------------
        # motion stream
        (self.feed('flow')
             .conv(7, 7, 96, 2, 2, name='flow_conv1')
             .max_pool(2, 2, 2, 2, padding='VALID', name='flow_pool1')
             .conv(5, 5, 256, 2, 2, name='flow_conv2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='flow_pool2')
             .conv(3, 3, 512, 1, 1, name='flow_conv3')
             .conv(3, 3, 512, 1, 1, name='flow_conv4')
             .conv(3, 3, 512, 1, 1, name='flow_conv5')
             .max_pool(2, 2, 2, 2, padding='VALID', name='flow_pool5')
             .fc(4096, name='flow_fc6')
             .fc(2048, name='flow_fc7')
             .make_time(name='time_feat')
             .fc(num_out=n_classes, name='flow_score', relu=False))

        (self.feed('space_score', 'flow_score')
            .combine_streams(name='cls_score')
            .softmax(name='cls_prob'))

        '''
        (self.feed('fc_new_2')
            .fc(num_out=4*n_classes, name='bbox_pred', relu=False))
        '''
        pass

    def retrieve_offsets(self):
        """ Retrieve deformable offsets defined in the network
        """
        offset_5a = self.get_output('res5a_branch2b_offset')
        offset_5b = self.get_output('res5b_branch2b_offset')
        offset_5c = self.get_output('res5c_branch2b_offset')
        return [offset_5a, offset_5b, offset_5c], \
               ['res5a_branch2b_offset',
                'res5b_branch2b_offset',
                'res5c_branch2b_offset']

    def compute_input_flow_gt(self, snippet_batch_):
        """Helper function to compute input groundtruth flow
        """
        return misc_utils.compute_optical_flow_sequence_batch(
            snippet_batch_, out_h=224, out_w=224)

    @layer
    def make_time(self, input, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            input_shape = input.shape.as_list()
            input_reshape = tf.reshape(
                input, [-1, self.snippet_len-self.max_time_gap]+input_shape[1:])
            input_reshape = tf.reduce_mean(input_reshape, axis=1)
        return input_reshape

    @layer
    def combine_streams(self, input, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            return (input[0] + input[1])*0.5
