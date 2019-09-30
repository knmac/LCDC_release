import tensorflow as tf
from .network import Network


class VGGnetTrain(Network):
    def build_graph(self):
        """ Build graph for model
        """
        n_classes = self.n_classes

        # Feature extraction
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

        # Deformable
        (self.feed('conv5_3')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
                  name='conv6_1_offset', padding='SAME', initializer='zeros'))
        (self.feed('conv5_3', 'conv6_1_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=True,
                         num_deform_group=4, name='conv6_1'))
        (self.feed('conv6_1')
            .conv(3, 3, 72, 1, 1, biased=True, rate=2, relu=False,
                  name='conv6_2_offset', padding='SAME', initializer='zeros'))
        (self.feed('conv6_1', 'conv6_2_offset')
            .deform_conv(3, 3, 512, 1, 1, biased=False, rate=2, relu=True,
                         num_deform_group=4, name='conv6_2'))

        # Classification
        (self.feed('conv6_2')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool6')
            .reshape(shape=(-1, 7, 7, 512), name='pool6_reshape')
            .fc(4096, name='fc6')
            .dropout(0.5, name='drop6')
            .fc(4096, name='fc7')
            .dropout(0.5, name='drop7')
            .fc(n_classes, relu=False, name='cls_score')
            .softmax(name='cls_prob'))
        pass

    def retrieve_offsets(self):
        """ Retrieve deformable offsets defined in the network
        """
        conv6_1_offset = self.get_output('conv6_1_offset')
        conv6_2_offset = self.get_output('conv6_2_offset')
        return [conv6_1_offset, conv6_2_offset], \
               ['conv6_1_offset', 'conv6_2_offset']
