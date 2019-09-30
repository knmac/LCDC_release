"""LCDC network
"""
from .network import Network, layer
import tensorflow as tf


class Resnet50_MotionConstraint(Network):
    @layer
    def expand_cir_shift(self, input, kh, kw, sh, sw, padding, name, reuse=False):
        """ Expand offset by circular shifting
        """
        _, h, w, c = input.shape.as_list()
        assert c == 2, 'only support input with 2 channels'
        assert padding == 'SAME', 'only support SAME padding now'
        assert sh == 1 and sw == 1, 'only support stride 1 now'
        assert kh == 3 and kw == 3, 'only support kernel size 3 now'

        with tf.variable_scope(name, reuse=reuse) as scope:
            # Pad the input for `SAME` padding
            # TODO: implement for `VALID` case
            # TODO: implement for kernel size != 3
            # TODO: implement for stride != 1
            paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            inp_pad = tf.pad(input, paddings)

            # Collect offsets
            output_lst = []  # list of output rows
            for i in range(1, h+1):
                output_col_lst = []  # list of output columns, at that row
                for j in range(1, w+1):
                    region = inp_pad[:, i-1:i+2, j-1:j+2, :]
                    output_col_lst.append(tf.reshape(region, (-1, kh*kw*c)))
                output_lst.append(tf.stack(output_col_lst, axis=1))
            output = tf.stack(output_lst, axis=1)
        _, oh, ow, oc = output.shape.as_list()
        assert oh == h and ow == w and oc == kh*kw*c, \
            'Need (?,{},{},{}), received (?,{},{},{})'.format(h, w, kh*kw*c,
                                                              oh, ow, oc)
        return output

    def build_graph(self):
        """ Build graph for model
        """
        n_classes = self.n_classes

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

        # Deformable ----------------------------------------------------------
        (self.feed('res4f_relu')
            .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5a_branch1'))

        (self.feed('res4f_relu')
            .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a',
                  padding='VALID')
            .batch_normalization(relu=False, name='bn5a_branch2a')
            .relu(name='res5a_branch2a_relu')
            .conv(3, 3, 2, 1, 1, biased=True, rate=2, relu=False,
                  name='res5a_branch2b_offset', padding='SAME', initializer='zeros')
            .expand_cir_shift(3, 3, 1, 1, 'SAME', name='res5a_branch2b_offset_expand'))

        (self.feed('res5a_branch2a_relu', 'res5a_branch2b_offset_expand')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=False,
                         num_deform_group=1, name='res5a_branch2b')
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
            # .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
            .conv(3, 3, 2, 1, 1, biased=True, rate=2, relu=False,
                  name='res5b_branch2b_offset', padding='SAME', initializer='zeros')
            .expand_cir_shift(3, 3, 1, 1, 'SAME', name='res5b_branch2b_offset_expand'))

        (self.feed('res5b_branch2a_relu', 'res5b_branch2b_offset_expand')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=False,
                         num_deform_group=1, name='res5b_branch2b')
                         # num_deform_group=4, name='res5b_branch2b')
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
            # .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
            .conv(3, 3, 2, 1, 1, biased=True, rate=2, relu=False,
                  name='res5c_branch2b_offset', padding='SAME', initializer='zeros')
            .expand_cir_shift(3, 3, 1, 1, 'SAME', name='res5c_branch2b_offset_expand'))

        (self.feed('res5c_branch2a_relu', 'res5c_branch2b_offset_expand')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=False,
                         num_deform_group=1, name='res5c_branch2b')
                         # num_deform_group=4, name='res5c_branch2b')
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

        # ---------------------------------------------------------------------
        # compute rDeRF
        self.compute_rDeRF()

        # Combining spatial and temporal
        (self.feed('conv_new_1_relu')
            .make_spacetime(name='spacetime_fusion')
            .conv3d_preserve(4, 3, 3, 256, 1, 1, 1, biased=False, relu=False,
                             padding='SAME', reduce_time=False,
                             name='spacetime_conv1')
            .batch_normalization(relu=False, name='spacetime_bn1')
            .relu(name='spacetime_relu1')
            .max_pool3d(2, 1, 1, 2, 1, 1, padding='VALID',
                        name='spacetime_pool1')
            .conv3d_preserve(4, 3, 3, 256, 1, 1, 1, biased=False, relu=False,
                             padding='SAME', reduce_time=False,
                             name='spacetime_conv2')
            .batch_normalization(relu=False, name='spacetime_bn2')
            .relu(name='spacetime_relu2')
            .max_pool3d(2, 1, 1, 2, 1, 1, padding='VALID',
                        name='spacetime_pool2')
            .reduce_time(name='spacetime_reduce'))

        # Classification
        (self.feed('spacetime_reduce')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool_new')
            .reshape(shape=(-1, 7, 7, 256), name='pool_new_reshape')
            .fc(num_out=1024, name='fc_new_1')
            .fc(num_out=1024, name='fc_new_2'))

        (self.feed('fc_new_2')
            .fc(num_out=n_classes, name='cls_score', relu=False)
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
