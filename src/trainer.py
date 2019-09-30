from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import glob
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.contrib.framework import get_variables_to_restore

from data_utils import dataset_factory
from networks import networks_factory, networks_utils
from networks.networks_utils import colorize_tensor, viz_flow

flags = tf.app.flags
FLAGS = flags.FLAGS

# directories
flags.DEFINE_string('logdir', None, 'where to store logs')
flags.DEFINE_string('recorddir', None, 'where the tfrecords are stored')
flags.DEFINE_string('record_regex', None, 'regular expression to retrive records')

# data reader and processing parameters
flags.DEFINE_string('datasetname', None, 'name of the dataset')
flags.DEFINE_string('labels_fname', None, 'filename containing label description')
flags.DEFINE_integer('batch_size', 1, 'number of snippet per training batch')
flags.DEFINE_integer('num_threads', 8, 'number of preprocess threads')
flags.DEFINE_integer('min_after_dequeue', 500, 'min after dequeue')
flags.DEFINE_integer('label_offset', 0, 'labels start with 0 or 1 in dataset')
flags.DEFINE_integer('snippet_len', None, 'number of frames per snippet')
flags.DEFINE_integer('raw_height', 480, 'raw image height for decoding tfrecords')
flags.DEFINE_integer('raw_width', 640, 'raw image width for decoding tfrecords')
flags.DEFINE_integer('raw_channel', 3, 'raw image channel for decoding tfrecords')
flags.DEFINE_integer('target_height', 224, 'image height after preprocessing')
flags.DEFINE_integer('target_width', 224, 'image height after preprocessing')
flags.DEFINE_integer('max_to_keep', 500, 'number of maximum checkpoints to keep.'
                                         'None or 0 means keeping everything.')
flags.DEFINE_integer('saving_freq', 500, 'How many iterations before saving checkpoints.'
                                         '0 means no saving')
flags.DEFINE_boolean('has_bg_lbl', True, 'has background class or not. If'
                                         'True, the number of classes will be'
                                         'increased by 1 from the content of'
                                         '`labels_fname`')

# training parameters
flags.DEFINE_string('netname', None, 'name of the network')
flags.DEFINE_string('pretrained_model', None, 'path to the pretrained model')
flags.DEFINE_string('pretrained_ext', None, 'extension of pretrained model')
flags.DEFINE_integer('num_iter', None, 'max number of iterations')
flags.DEFINE_integer('decay_steps', None, 'number of iterations before decay')
flags.DEFINE_float('motion_scaling', 1.0, 'motion loss scaling factor')
flags.DEFINE_boolean('usemotionloss', True, 'If True, will include motion loss'
                                            'in total loss. False means the '
                                            'regular CNN network')
flags.DEFINE_boolean('initoffset', True, 'If True, will initialize offsets as Gaussian')


flags.DEFINE_float('init_learning_rate', 1e-4, 'initial learning rate')
flags.DEFINE_float('decay_rate', 0.96, 'decaying rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_boolean('resume', False, 'resume training process. Will load'
                                      'pretrained model if set as False')
flags.DEFINE_string('optimizer', 'Momentum', 'Adam or Momentum')
flags.DEFINE_string('wrapper', 'manual', 'training wrapper')
flags.DEFINE_integer('max_time_gap', 1, 'maximum time gap for motion loss')
flags.DEFINE_string('gap_scales', None, 'scaling for each temporal gap, string'
                                        'of floats separated by commas, e.g'
                                        '`0.75,0.4` means 0.75 for gap1 and'
                                        '0.4 for gap2. If left as None, all'
                                        'scales are set as 1.')

flags.DEFINE_boolean('_DEBUG_', False, 'debug mode')
_DEBUG_ = FLAGS._DEBUG_


def check_args():
    """Check input arguments before running.
    """
    assert os.path.exists(FLAGS.recorddir)
    assert FLAGS.optimizer == 'Adam' or FLAGS.optimizer == 'Momentum'
    assert FLAGS.wrapper == 'supervisor' or FLAGS.wrapper == 'manual'
    pass


def train_step(sess, total_loss, global_step, metrics_op, train_op, lr,
               accuracy_vid, feed_dict=None):
    """Train step function.

    Run training operation and and log results if necessary
    """
    # check time for each sess run
    start_time = time.time()
    if feed_dict is not None:
        total_loss_, global_step_, _, _ = sess.run(
            [total_loss, global_step, metrics_op, train_op],
            feed_dict=feed_dict)
    else:
        # experimental
        total_loss_, global_step_, _, _ = sess.run(
            [total_loss, global_step, metrics_op, train_op])
    time_elapsed = time.time() - start_time

    # print results
    if global_step_ % 100 == 0:
        tf.logging.info(
            'global step %s: total_loss: %.3f (%.3f sec/step)',
            global_step_, total_loss_, time_elapsed)

    if global_step_ % 1000 == 0:
        lr_, accuracy_vid_ = sess.run([lr, accuracy_vid])
        tf.logging.info('Learning rate: %s', lr_)
        tf.logging.info('Streaming accuracy on video: %s', accuracy_vid_)
    return total_loss_


def solver_supervisor(net, init_fn, snippet_batch, label_batch, total_loss,
                      global_step, metrics_op, train_op, lr, accuracy_vid,
                      summary_op):
    """Train network with supervisor wrapper (experimental)"""
    # create supervisor
    sv = tf.train.Supervisor(
        logdir=FLAGS.logdir, summary_op=None, init_fn=init_fn)

    # create session
    with sv.managed_session() as sess:
        start_step = sess.run(global_step)
        for step in range(start_step, start_step+FLAGS.num_iter):
            if sv.should_stop():
                break

            # produce inputs
            snippet_batch_, labels_batch_ = sess.run(
                [snippet_batch, label_batch])
            feed_dict = {net.data_raw: snippet_batch_,
                         net.labels_raw: labels_batch_}

            # training
            loss_ = train_step(sess, total_loss, global_step,
                               metrics_op, train_op, lr, accuracy_vid,
                               feed_dict)
            if _DEBUG_:
                print(step, loss_)

            # log the summaries manually
            if step % 10 == 0:
                summary_ = sess.run(summary_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summary_)

        # log final loss and accuracy
        tf.logging.info('Final loss: %s', loss_)
        tf.logging.info('Final accuracy on video: %s', sess.run(accuracy_vid))
        tf.logging.info('Finished training! Saving model to disk now.')
        sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
    pass


def solver_manual(net, init_fn, snippet_batch, label_batch, total_loss,
                  global_step, metrics_op, train_op, lr, accuracy_vid,
                  summary_op):
    """Train network with manually created coordinator and logger
    """
    with tf.Session() as sess:
        # initialization routine
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        init_fn(sess)

        trainables = tf.trainable_variables()
        print('\n'+'='*80)
        print('Verifying initialization weights')
        for x in trainables:
            x_val = sess.run(x)
            print('{} \t mean={:05f} var={:05f} min={:05f} max={:05f}'.format(
                x.op.name, np.mean(x_val), np.var(x_val),
                np.min(x_val), np.max(x_val)))
            if not np.any(x_val):
                print('--> All zeros')

        # summary and checkpoint managers
        sum_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        ckpt_saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

        # coordinator for input producer
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # iterations of training phase
        tf.logging.info('Begin training...')
        start_step = sess.run(global_step)
        for step in range(start_step, FLAGS.num_iter):
            if coord.should_stop():
                break

            # produce inputs
            snippet_batch_, labels_batch_ = sess.run(
                [snippet_batch, label_batch])
            if hasattr(net, 'use_flow'):
                flow_batch_ = net.compute_input_flow_gt(snippet_batch_)
                feed_dict = {net.data_raw: snippet_batch_,
                             net.labels_raw: labels_batch_,
                             net.flow_gt: flow_batch_}
            else:
                feed_dict = {net.data_raw: snippet_batch_,
                             net.labels_raw: labels_batch_}

            # training
            loss_ = train_step(sess, total_loss, global_step, metrics_op,
                               train_op, lr, accuracy_vid, feed_dict)
            if _DEBUG_:
                print(step, loss_)

            # log the summaries manually
            if step % 10 == 0:
                summary_, global_step_ = sess.run(
                    [summary_op, global_step], feed_dict=feed_dict)
                sum_writer.add_summary(summary_, global_step_)

            # save checkpoint
            # if step % 500 == 0:
            if (FLAGS.saving_freq != 0) and (step % FLAGS.saving_freq == 0):
                ckpt_saver.save(
                    sess, os.path.join(FLAGS.logdir, 'model.ckpt'),
                    global_step=sess.run(global_step))

        # log final loss and accuracy
        tf.logging.info('Final loss: %s', loss_)
        tf.logging.info('Final accuracy on video: %s', sess.run(accuracy_vid))
        tf.logging.info('Finished training! Saving model to disk now.')
        ckpt_saver.save(
            sess, os.path.join(FLAGS.logdir, 'model.ckpt'),
            global_step=sess.run(global_step))

        # close input producer
        coord.request_stop()
        coord.join(threads)
    pass


def main(_):
    """Main function"""
    # check input arguments
    check_args()

    # retrieve labels
    lbl_list = open(FLAGS.labels_fname).read().splitlines()
    n_classes = len(lbl_list)
    if FLAGS.has_bg_lbl:
        n_classes += 1

    # retrieve the list of tfrecord files
    tfrecord_list = glob.glob(os.path.join(FLAGS.recorddir,
                                           FLAGS.record_regex))
    tfrecord_list.sort()

    # set verbosity for info level only
    tf.logging.set_verbosity(tf.logging.INFO)

    # contruct graph and build models
    with tf.Graph().as_default():
        # get input producer
        raw_shape = [FLAGS.raw_height, FLAGS.raw_width, FLAGS.raw_channel]
        dataset_reader = dataset_factory.get_reader(FLAGS.datasetname)
        reader = dataset_reader(FLAGS.batch_size, FLAGS.num_threads,
                                FLAGS.min_after_dequeue, raw_shape=raw_shape,
                                target_height=FLAGS.target_height,
                                target_width=FLAGS.target_width,
                                label_offset=FLAGS.label_offset)
        snippet_batch, label_batch = reader.read_snippet_record(
            tfrecord_list, FLAGS.snippet_len, shuffle_fname_queue=True,
            is_training=True)

        # build network
        net = networks_factory.build_net(
            FLAGS.netname, n_classes, FLAGS.snippet_len,
            FLAGS.target_height, FLAGS.target_width,
            max_time_gap=FLAGS.max_time_gap,
            trainable=True)

        # compute losses
        total_loss, loss_dict = net.compute_total_loss(
            usemotionloss=FLAGS.usemotionloss,
            scaling_factor=FLAGS.motion_scaling,
            gap_scales=FLAGS.gap_scales)
        # total_loss, class_loss, motion_loss = losses
        try:
            pyramid_losses = net.get_pyramid_class_losses()
        except AttributeError:
            pyramid_losses = None

        # metrics to predict
        (accuracy_vid, confusion_vid, metrics_op) = net.create_metrics()
        if isinstance(confusion_vid, list):
            confusion_vid_img = [colorize_tensor(x, extend=True) for x in confusion_vid]
        else:
            confusion_vid_img = colorize_tensor(confusion_vid, extend=True)

        # setup optimizer and training operation
        global_step = get_or_create_global_step()
        lr = tf.train.exponential_decay(
            learning_rate=FLAGS.init_learning_rate,
            global_step=global_step,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.decay_rate,
            staircase=True)
        if FLAGS.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        elif FLAGS.optimizer == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr,
                                                   momentum=FLAGS.momentum)
        train_op = optimizer.minimize(total_loss, global_step=global_step)

        # summaries of losses and metrics
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('losses/total_loss', total_loss)
        for loss_type in loss_dict.keys():
            if loss_dict[loss_type] is not None:
                tf.summary.scalar('losses/'+loss_type, loss_dict[loss_type])
        # tf.summary.scalar('losses/class_loss', class_loss)
        # tf.summary.image('data_samples', snippet_batch[0], max_outputs=3)
        # if motion_loss is not None:
            # tf.summary.scalar('losses/motion_loss', motion_loss)
        if pyramid_losses is not None:
            for scale_loss in pyramid_losses:
                tf.summary.scalar('losses/'+scale_loss.op.name, scale_loss)

        if isinstance(accuracy_vid, list):
            for item in accuracy_vid:
                tf.summary.scalar('accuracy/'+item.op.name, item)
        else:
            tf.summary.scalar('accuracy', accuracy_vid)

        if isinstance(confusion_vid_img, list):
            for item in confusion_vid_img:
                tf.summary.image('confusion/'+item.op.name, item)
        else:
            tf.summary.image('confusion', confusion_vid_img)

        # summaries of rDeRF
        rDeRF_dict = net.rDeRF_dict
        if rDeRF_dict != {}:
            offset_list_frame = net.offset_list_frame
            offset_name_list = net.offset_name_list
            n_resolutions = len(rDeRF_dict['gap1'])
            for gap in range(1, FLAGS.max_time_gap+1):
                gap_key = 'gap' + str(gap)
                rDeRF_list = rDeRF_dict[gap_key]
                for i in range(n_resolutions):
                    offset_name = offset_name_list[i]
                    tf.summary.histogram(gap_key+'_rDeRF/'+offset_name,
                                         rDeRF_list[i])
                    tf.summary.scalar(gap_key+'_mean_rDeRF/'+offset_name,
                                      tf.reduce_mean(rDeRF_list[i]))
            for i in range(n_resolutions):
                offset_name = offset_name_list[i]
                tf.summary.histogram('offset/'+offset_name,
                                     offset_list_frame[i])
                tf.summary.scalar('mean_offset/'+offset_name,
                                  tf.reduce_mean(offset_list_frame[i]))

        if hasattr(net, 'use_flow') and net.use_flow:
            assert FLAGS.max_time_gap == 1, 'only implemented for gap=1'
            tf.summary.image('flow_gt', viz_flow(net.flow_gt[0][0]))
            n_resolutions = len(rDeRF_dict['gap1'])
            for i in range(n_resolutions):
                offset_name = offset_name_list[i]
                tf.summary.image(offset_name, viz_flow(rDeRF_list[i][0][0]))
        summary_op = tf.summary.merge_all()

        # restore pretrained model or resume training when initializing net
        if FLAGS.resume:
            variables_to_restore = get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            ckpt_fname = tf.train.latest_checkpoint(FLAGS.logdir)

            # try to manually find checkpoint
            if ckpt_fname is None:
                ckpt_tmp = glob.glob(os.path.join(FLAGS.logdir, 'model.ckpt*'))
                assert ckpt_tmp != [], 'Found no checkpoints'
                ckpt_tmp.sort()
                ckpt_fname = ckpt_tmp[-1]
                ckpt_fname = ckpt_fname[:ckpt_fname.rfind('.')]

            def init_fn(sess):
                tf.logging.info('Resuming training process')
                return saver.restore(sess, ckpt_fname)
        else:
            assign_ops = networks_utils.load_pretrained(
                FLAGS.pretrained_model, ignore_missing=True,
                extension=FLAGS.pretrained_ext,
                initoffset=FLAGS.initoffset)

            def init_fn(sess):
                tf.logging.info('Loading pretrained weights')
                sess.run(assign_ops)

        # training phase
        if FLAGS.wrapper == 'manual':
            solver_manual(net, init_fn, snippet_batch, label_batch, total_loss,
                          global_step, metrics_op, train_op, lr, accuracy_vid,
                          summary_op)
        elif FLAGS.wrapper == 'supervisor':
            solver_supervisor(net, init_fn, snippet_batch, label_batch,
                              total_loss, global_step, metrics_op, train_op,
                              lr, accuracy_vid, summary_op)
    pass


if __name__ == '__main__':
    tf.app.run()
