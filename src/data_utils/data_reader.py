import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class DataReader(object):
    def __init__(self, batch_size, num_threads, min_after_dequeue,
                 raw_shape, target_height=None, target_width=None,
                 label_offset=0):
        """
        Args:
            batch_size: number of samples per batch
            num_threads: number of reading threads
            min_after_dequeue: minimum samples after dequeuing
            raw_shape: raw image shape
            target_height: new image height after resizing, None mean no resizing
            target_width: new image width after resizing, None mean no resizing
            label_offset: the first label index
        """
        assert len(raw_shape) == 3

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.min_after_dequeue = min_after_dequeue
        self.raw_shape = raw_shape
        if target_height is None:
            self.target_height = raw_shape[0]
        else:
            self.target_height = target_height
        if target_width is None:
            self.target_width = raw_shape[1]
        else:
            self.target_width = target_width
        self.label_offset = label_offset

        self.capacity = min_after_dequeue + 3*batch_size
        pass

    def serialize_data(self, tfrecord_list, to_shuffle):
        """ Produce serialized data from a list of tfrecord_list

        Args:
            tfrecord_list: list of tfrecord files to read
            to_shuffle: boolean, whether filename queues should be shuffled

        Returns:
            serialized_example: next record value produced by the reader
        """
        assert type(tfrecord_list) is list
        assert type(to_shuffle) is bool

        # make filename queue
        fname_queue = tf.train.string_input_producer(
                tfrecord_list, shuffle=to_shuffle)

        # define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(fname_queue)
        return serialized_example

    def produce_batch(self, data, is_training):
        """ Enqueue data and produce batches of tensors

        Args:
            data: a list of tensors to enqueue
            is_training: boolean, if True then the batches are shuffled, 
                         otherwise the batches are not and smaller final batch
                         is allowed

        Returns
            data_batch: a list of tensors, corresponding to the items in data
        """
        assert type(data) is list
        assert type(is_training) is bool

        # Creates batches by randomly shuffling tensors
        if is_training:
            data_batch = tf.train.shuffle_batch(
                    data, batch_size=self.batch_size,
                    capacity=self.capacity, num_threads=self.num_threads,
                    min_after_dequeue=self.min_after_dequeue)
        else:
            data_batch = tf.train.batch(
                    data, batch_size=self.batch_size,
                    capacity=self.capacity, num_threads=self.num_threads,
                    allow_smaller_final_batch=True)
        return data_batch
