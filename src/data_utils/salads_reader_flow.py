"""tfrecord reader for 50 salads dataset, with flow data
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
from data_utils.data_reader import DataReader
from data_utils.batch_preprocessing import random_crop, central_crop


class SaladsReaderFlow(DataReader):
    def _process_flow(self, data, is_training):
        """Simple process flow data by cropping

        Args:
            data: tensor of shape (h, w, 2) or (n, h, w, 2)
            is_training: whether during training phase

        Returns:
            processed tensor
        """
        # get the shape of input
        assert data.shape.ndims == 3 or data.shape.ndims == 4, \
            'Only allows tensors of 3 or 4 dimensions'
        if data.shape.ndims == 3:
            h, w, c = data.shape.as_list()
        else:
            _, h, w, c = data.shape.as_list()
        assert c == 2, 'Only allows tensors with 2 channels'

        if is_training:
            data = random_crop(data, self.target_height, self.target_width)
        else:
            data = central_crop(data, self.target_height, self.target_width)
        return tf.cast(data, tf.float32)

    def read_img_record(self, tfrecord_list, shuffle_fname_queue, is_training):
        """ Read image record and produce batches of images and labels

        Args:
            tfrecord_list: list of tfrecord files
            shuffle_fname_queue: whether to shuffle the provided tfrecord files
            is_training: whether during training phase

        Returns:
            im_batch: tensor, batch of images
            lbl_batch: tensor, batch of labels
        """
        # Serialize the list of tfrecord files
        serialized_example = DataReader.serialize_data(
            self, tfrecord_list, shuffle_fname_queue)

        # Define feature structure
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example,
                                           features=feature)

        # Process data
        label = tf.cast(features['label'], tf.int32) - self.label_offset
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, self.raw_shape)
        image = self._process_flow(image, is_training)

        # Produce data batch and label batch
        im_batch, lbl_batch = DataReader.produce_batch(
            self, [image, label], is_training)
        return im_batch, lbl_batch

    def read_snippet_record(self, tfrecord_list, snippet_len,
                            shuffle_fname_queue, is_training):
        """ Read snippet record and produce batches of snippets and labels

        Args:
            tfrecord_list: list of tfrecord files
            snippet_len: number of frames per snippet
            shuffle_fname_queue: whether to shuffle the provided tfrecord files
            is_training: whether during training phase

        Returns:
            snippet_batch: tensor, batch of snippets
            lbl_batch: tensor, batch of labels
        """
        # Serialize the list of tfrecord files
        serialized_example = DataReader.serialize_data(
            self, tfrecord_list, shuffle_fname_queue)

        # Define feature structure
        feature = {'snippet': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example,
                                           features=feature)

        # Process data
        label = tf.cast(features['label'], tf.int32) - self.label_offset
        snippet = tf.decode_raw(features['snippet'], tf.float32)
        snippet_shape = [snippet_len] + self.raw_shape
        snippet = tf.reshape(snippet, snippet_shape)
        snippet = self._process_flow(snippet, is_training)

        # Produce data batch and label batch
        snippet_batch, lbl_batch = DataReader.produce_batch(
            self, [snippet, label], is_training)
        return snippet_batch, lbl_batch
