"""VGG for optical flow. Follow the implementation of 2 stream network.

Input shape is (?, snippet_len, height, width, 2) and will be reshaped to
(?, height, width, snippet_len*2) before feeding in.
"""
from .network import Network, layer
import tensorflow as tf
from data_utils import misc_utils


class VGGVanillaOpticalFlowV2(Network):
    def __init__(self, n_classes, snippet_len, height, width, max_time_gap,
                 trainable):
        self.snippet_len = snippet_len
        self.n_classes = n_classes
        self.keep_prob = tf.placeholder(tf.float32)
        self.trainable = trainable
        self.max_time_gap = max_time_gap
        assert max_time_gap >= 0 and max_time_gap < self.snippet_len, \
            'invalid range of max_time_gap: {}'.format(max_time_gap)

        # placeholders for rDeRF and related variables
        self.rDeRF_dict = {}
        self.offset_list_frame = []
        self.offset_list_vid = []
        self.offset_name_list = []

        # placeholders for input data and labels
        self.data_raw = tf.placeholder(
            shape=[None, self.snippet_len, height, width, 2],
            name='data_raw',
            dtype=tf.float32)
        self.labels_raw = tf.placeholder(
            shape=[None],
            name='labels_raw',
            dtype=tf.int32)

        # Convert shape from (?, N, H, W, 2) to (?, H, W, 2*N)
        # # Transpose from (?, N, H, W, 2) to (?, 2, N, H, W) first
        tmp = tf.transpose(self.data_raw, perm=[0, 4, 1, 2, 3])
        # # Then reshape from (?, 2, N, H, W) to (?, 2N, H, W)
        tmp = tf.reshape(tmp, [-1, 2*snippet_len, height, width])
        # # Finally, transpose from (?, 2N, H, W) to (?, H, W, 2N)
        self.data = tf.reshape(
            tmp, [-1, height, width, 2*snippet_len],
            name='data')

        # tiling labels for multiple frames of the same video
        # [batch_size] -> [batch_size, snippet_len]
        labels_expand = tf.expand_dims(self.labels_raw, axis=1)
        labels_tile = tf.tile(labels_expand, [1, snippet_len])
        self.labels = tf.reshape(labels_tile, [-1], name='labels')

        self.layers = dict({'data': self.data})
        self.build_graph()

        pass

    def build_graph(self):
        """ Build graph for model
        """
        n_classes = self.n_classes

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))

        self.compute_rDeRF()  # dummy

        # Classification
        (self.feed('conv5_3')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool6')
            .reshape(shape=(-1, 7, 7, 512), name='pool6_reshape')
            .fc(4096, name='fc6')
            .dropout(0.5, name='drop6')
            .fc(4096, name='fc7')
            .dropout(0.5, name='drop7')
            # .make_time(name='drop7_reduced')
            .fc(n_classes, relu=False, name='cls_score')
            .softmax(name='cls_prob'))
        pass

    def retrieve_offsets(self):
        """ Retrieve deformable offsets defined in the network
        """
        return [], []
