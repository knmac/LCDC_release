"""Check model's complexity by counting the number of trainable parameters
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
import numpy as np

from networks import networks_factory

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('netname', None, 'name of the network')
flags.DEFINE_string('labels_fname', None, 'filename containing label description')
flags.DEFINE_boolean('has_bg_lbl', True, 'has background class or not. If'
                                         'True, the number of classes will be'
                                         'increased by 1 from the content of'
                                         '`labels_fname`')
flags.DEFINE_integer('snippet_len', None, 'number of frames per snippet')
flags.DEFINE_integer('target_height', 224, 'image height after preprocessing')
flags.DEFINE_integer('target_width', 224, 'image height after preprocessing')
flags.DEFINE_integer('max_time_gap', 1, 'maximum time gap for motion loss')


def main(_):
    """Main function"""
    # retrieve labels
    lbl_list = open(FLAGS.labels_fname).read().splitlines()
    n_classes = len(lbl_list)
    if FLAGS.has_bg_lbl:
        n_classes += 1

    networks_factory.build_net(
        FLAGS.netname, n_classes, FLAGS.snippet_len,
        FLAGS.target_height, FLAGS.target_width,
        max_time_gap=FLAGS.max_time_gap,
        trainable=True)

    print('-'*80)
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.shape
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print(variable.op.name, shape, '-->', variable_parameters)
        total_parameters += variable_parameters

    print('\nTotal number of parameters:', total_parameters)
    pass


if __name__ == '__main__':
    tf.app.run()
