"""tfrecord reader for 50 salads dataset
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
import numpy as np
from data_utils.data_reader import DataReader
from data_utils import batch_preprocessing


class SaladsReader(DataReader):
    def _no_augment_process_fn(self, data):
        """ Preprocess an image or a snippet with mean removal and resizing.
        The aspect ratio is kept by zero-padding.

        Args:
            img: tensor of shape (h, w, 3) or (n, h, w, 3)

        Returns:
            processed tensor
        """
        # get the shape of input
        assert data.shape.ndims == 3 or data.shape.ndims == 4, \
                'Only allows tensors of 3 or 4 dimensions'
        if data.shape.ndims == 3:
            h, w, _ = data.shape.as_list()
        else:
            _, h, w, _ = data.shape.as_list()

        # resize data so that the largest size is the same as target size
        if h < w:
            new_w = self.target_width
            scale = new_w * 1.0 / w
            new_h = int(scale * h)
        else:
            new_h = self.target_height
            scale = new_h * 1.0 / h
            new_w = int(scale * w)
        data = tf.image.resize_images(data, [new_h, new_w])

        # pad the image with zeros. Since we already resized the inputs wrt
        # the largest side, this operation only pads zeros
        data = tf.image.resize_image_with_crop_or_pad(
                data, self.target_height, self.target_width)

        # mean removal
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        mean_img = np.zeros(data.shape.as_list())

        mean_img[..., 0] = _R_MEAN
        mean_img[..., 1] = _G_MEAN
        mean_img[..., 2] = _B_MEAN

        return tf.cast(data, tf.float32) - mean_img

    def read_img_record(self, tfrecord_list, shuffle_fname_queue, is_training,
                        augment=True):
        """ Read image record and produce batches of images and labels

        Args:
            tfrecord_list: list of tfrecord files
            shuffle_fname_queue: whether to shuffle the provided tfrecord files
            is_training: whether during training phase
            augment: whether to augment data

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
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, self.raw_shape)
        if augment:
            image = batch_preprocessing.preprocess(
                image, self.target_height, self.target_width, is_training)
        else:
            image = self._no_augment_process_fn(image)

        # Produce data batch and label batch
        im_batch, lbl_batch = DataReader.produce_batch(
                self, [image, label], is_training)
        return im_batch, lbl_batch

    def read_snippet_record(self, tfrecord_list, snippet_len,
                            shuffle_fname_queue, is_training, augment=True):
        """ Read snippet record and produce batches of snippets and labels

        Args:
            tfrecord_list: list of tfrecord files
            snippet_len: number of frames per snippet
            shuffle_fname_queue: whether to shuffle the provided tfrecord files
            is_training: whether during training phase
            augment: whether to augment data

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
        snippet = tf.decode_raw(features['snippet'], tf.uint8)
        snippet_shape = [snippet_len] + self.raw_shape
        snippet = tf.reshape(snippet, snippet_shape)
        if augment:
            snippet = batch_preprocessing.preprocess(
                snippet, self.target_height, self.target_width, is_training)
        else:
            snippet = self._no_augment_process_fn(snippet)

        # Produce data batch and label batch
        snippet_batch, lbl_batch = DataReader.produce_batch(
                self, [snippet, label], is_training)
        return snippet_batch, lbl_batch
