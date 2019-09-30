"""Extract features and save as .mat files for ED-TCN. Only used for motion
stream (e.g. vgg_vanilla_opticalflow_v2)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import tensorflow as tf
import skimage.io
import cv2

from progressbar import ProgressBar
from data_utils import dataset_factory
from networks import networks_factory, networks_utils
from tensorflow.contrib.framework import get_variables_to_restore

from extract4tcn import make_mat_file  # reuse the FLAGS from this

flags = tf.app.flags
FLAGS = flags.FLAGS


def read_n_compute_flow(im_batch):
    """Read images from given paths then compute optical flow
    """
    # Allocate memory
    flows = np.zeros([len(im_batch), FLAGS.target_height, FLAGS.target_width, 2],
                     dtype=np.float32)

    # Load images
    for i in range(len(im_batch) - 1):
        im1 = skimage.io.imread(im_batch[i])
        im2 = skimage.io.imread(im_batch[i+1])

        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        im1 = skimage.transform.resize(
            im1, [FLAGS.target_height, FLAGS.target_width], preserve_range=True,
            mode='constant', anti_aliasing=True)
        im2 = skimage.transform.resize(
            im2, [FLAGS.target_height, FLAGS.target_width], preserve_range=True,
            mode='constant', anti_aliasing=True)
        flow = cv2.calcOpticalFlowFarneback(
            im1, im2, flow=None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Store images
        flows[i] = flow

    # Replicate the flow for last frame
    flows[-1] = flow
    return flows


def main(_):
    """Main function"""
    # Make outputdir
    if not os.path.exists(FLAGS.outputdir):
        os.makedirs(FLAGS.outputdir)

    # Load video list
    vid_lst = os.listdir(FLAGS.segmented_dir)
    vid_lst.sort()

    # Load label dictionary
    lbl_list = open(FLAGS.lbl_dict_pth).read().splitlines()
    n_classes = len(lbl_list)
    if FLAGS.has_bg_lbl:
        n_classes += 1

    # Use the load_snippet_pths_test in data writer to get frames and labels
    dataset_writer = dataset_factory.get_writer(FLAGS.datasetname)
    writer = dataset_writer()

    # set default graph
    with tf.Graph().as_default():
        # build network
        net = networks_factory.build_net(
            FLAGS.netname, n_classes, FLAGS.snippet_len,
            FLAGS.target_height, FLAGS.target_width,
            max_time_gap=FLAGS.max_time_gap,
            trainable=False)

        # extract features
        feat = net.get_output(FLAGS.featname)

        # load pretrained weights
        if '.pkl' in FLAGS.pretrained_model:
            assign_ops = networks_utils.load_pretrained(
                FLAGS.pretrained_model, ignore_missing=True,
                extension='pkl',
                initoffset=FLAGS.usemotionloss)
        else:
            variables_to_restore = get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            def init_fn(sess):
                tf.logging.info('Restoring checkpoint...')
                return saver.restore(sess, FLAGS.pretrained_model)

        # create session
        with tf.Session() as sess:
            # initialization
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            if '.pkl' in FLAGS.pretrained_model:
                sess.run(assign_ops)
            else:
                init_fn(sess)

            # for each video in video list
            n_vids = len(vid_lst)
            snippet_len = FLAGS.snippet_len + 1  # original snippet_len without flow

            for vid_id in range(n_vids):
                # skip existing feature files
                output_fname = '{}.avi.mat'.format(vid_lst[vid_id])
                if os.path.exists(os.path.join(FLAGS.outputdir, output_fname)):
                    print('{} already exists'.format(output_fname))
                    continue

                # load all file names and labels
                vid = vid_lst[vid_id]
                print('\nExtracting features for ' + vid)
                fname_lst, lbl_lst = writer.load_snippet_pths_test(
                    FLAGS.segmented_dir, [vid], FLAGS.lbl_dict_pth,
                    FLAGS.bg_lbl, FLAGS.ext, FLAGS.frameskip)
                fname_lst = [x[0] for x in fname_lst]

                # prefetch all frames of a video
                frames_all = read_n_compute_flow(fname_lst)

                # prepare indices
                n_frames = len(lbl_lst)
                left = snippet_len // 2  # correct, because snippet_len was increased by 1
                right = snippet_len - left

                # go through the video frames in acausal fashion
                frame_id = left
                feats_per_vid = []
                groundtruths_per_vid = []
                pbar = ProgressBar(max_value=n_frames)
                while frame_id < n_frames-right+1:
                    # produce inputs
                    snippet_batch = []
                    lbl_batch = []
                    for _ in range(FLAGS.batch_size):
                        if frame_id+right > n_frames:
                            break

                        # ignore 1 last frame because this is flow
                        snippet = frames_all[frame_id-left:frame_id+right - 1]
                        # this is in sync with label from appearance stream
                        lbl = lbl_lst[frame_id]

                        snippet_batch.append(snippet)
                        lbl_batch.append(lbl)
                        frame_id += FLAGS.stride

                    feed_dict = {net.data_raw: snippet_batch,
                                 net.labels_raw: lbl_batch}

                    # extract features
                    feat_ = sess.run(feat, feed_dict=feed_dict)

                    # append data
                    for i in range(feat_.shape[0]):
                        feats_per_vid.append(feat_[i])
                        groundtruths_per_vid.append(lbl_batch[i])
                    pbar.update(frame_id)

                # produce mat file for a video
                feats_per_vid = np.array(feats_per_vid, dtype=np.float32)
                groundtruths_per_vid = np.array(groundtruths_per_vid)
                make_mat_file(output_fname, feats_per_vid,
                              groundtruths_per_vid,
                              expected_length=n_frames//FLAGS.stride)
            pass
        pass
    pass


if __name__ == '__main__':
    tf.app.run()
