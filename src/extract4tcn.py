"""Extract features and save as .mat files for ED-TCN. Only used for
spatial-temporal or appearance stream (in the case of 2 stream). Do NOT use
for motion stream.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import tensorflow as tf
import skimage.io as sio
import scipy.io

from skimage.transform import resize
from progressbar import ProgressBar
from data_utils import dataset_factory
from networks import networks_factory, networks_utils
from tensorflow.contrib.framework import get_variables_to_restore

flags = tf.app.flags
FLAGS = flags.FLAGS

# paths and directories
flags.DEFINE_string('segmented_dir', None,
                    'segmented frames, used for reference')
flags.DEFINE_string('pretrained_model', None,
                    'path to the pretrained model')
flags.DEFINE_string('lbl_dict_pth', None,
                    'path to label dictionary')
flags.DEFINE_string('outputdir', None,
                    'output directory')
flags.DEFINE_string('featname', None,
                    'name of the layer to extract features')

# other parameters
flags.DEFINE_string('datasetname', '50salads', 'name of the dataset')
flags.DEFINE_integer('frameskip', 5, 'frame skip for downsampling')
flags.DEFINE_integer('stride', 2, 'stride after downsampling (this is testing '
                                  'stride, not training stride)')
flags.DEFINE_string('netname', None, 'Resnet50 without offsets')
flags.DEFINE_string('bg_lbl', 'background', 'name of the background class')
flags.DEFINE_string('ext', 'png', 'extension of frame file names')
flags.DEFINE_integer('snippet_len', 1, 'extract features frame by frame')
flags.DEFINE_integer('target_height', 224, 'target image height')
flags.DEFINE_integer('target_width', 224, 'target image width')
flags.DEFINE_integer('batch_size', 1, 'number of images to feed at a time')
flags.DEFINE_integer('max_time_gap', 1, 'maximum time gap for motion loss')
flags.DEFINE_boolean('usemotionloss', False, 'no need to use motion loss')
flags.DEFINE_boolean('has_bg_lbl', True, 'has background class or not. If'
                                         'True, the number of classes will be'
                                         'increased by 1 from the content of'
                                         '`labels_fname`')
flags.DEFINE_boolean('use_single_mid', False, 'use a single middle frame. Used for vanilla')

# set up mean image
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
im_mean = np.array([_R_MEAN, _G_MEAN, _B_MEAN], dtype=np.float32)


def read_n_process(im_batch):
    """Read images from given path then preprocess them

    Args:
        im_batch: a list of image file names

    Returns:
        images: preprocessed images (central crop and mean removal)
    """
    # allocate memory
    target_shape = [FLAGS.target_height, FLAGS.target_width, 3]
    images = np.zeros([len(im_batch)] + target_shape, dtype=np.float32)

    # load each image
    for i in range(len(im_batch)):
        # load images from filenames
        img = sio.imread(im_batch[i])

        # resize image
        img = resize(img, (FLAGS.target_height, FLAGS.target_width, 3),
                     mode='constant', preserve_range=True)

        # mean removal
        img -= im_mean

        # append
        images[i] = img
    return images


def make_mat_file(output_fname, all_feat, lbl_lst, expected_length=None):
    """Create mat files from given feature and label list to match Lea's
    file format

    Args:
        all_feat: all extracted feature, ndarray (N, feat_dim)
        lbl_lst: list of all labels, length of N
    """
    # Expand or reduce the feature array if needed
    if expected_length is not None:
        N = all_feat.shape[0]
        if expected_length < N:
            all_feat = all_feat[:expected_length]
            lbl_lst = lbl_lst[:expected_length]
        elif expected_length > N:
            diff = expected_length - N
            left = np.ceil(diff / 2.0).astype(np.int)
            right = diff - left

            # Expand features
            left_feat = np.expand_dims(all_feat[0], axis=0)
            left_pad = np.repeat(left_feat, left, axis=0)

            right_feat = np.expand_dims(all_feat[-1], axis=0)
            right_pad = np.repeat(right_feat, right, axis=0)

            all_feat = np.concatenate([left_pad, all_feat, right_pad], axis=0)

            # Expand labels
            left_lbl = np.repeat(lbl_lst[0], left)
            right_lbl = np.repeat(lbl_lst[-1], right)
            lbl_lst = np.concatenate([left_lbl, lbl_lst, right_lbl])

    assert len(all_feat) == len(lbl_lst), \
        'features and labels list must have the same length'

    # Save as matlab *mat file
    mdict = {'A': all_feat,
             'Y': np.expand_dims(lbl_lst, axis=1)}
    scipy.io.savemat(os.path.join(FLAGS.outputdir, output_fname), mdict)
    pass


def main(_):
    """Main function"""
    if not os.path.exists(FLAGS.outputdir):
        os.makedirs(FLAGS.outputdir)

    # load video list
    vid_lst = os.listdir(FLAGS.segmented_dir)
    vid_lst.sort()

    # load label dictionary
    lbl_list = open(FLAGS.lbl_dict_pth).read().splitlines()
    n_classes = len(lbl_list)
    if FLAGS.has_bg_lbl:
        n_classes += 1
        # lbl_dict = {'background': 0}
        # for i in range(len(lbl_list)):
        #     lbl_dict[lbl_list[i]] = i + 1
        #     lbl_dict[lbl_list[i]] = i

    # use the load_snippet_pths_test in data writer to get frames and labels
    dataset_writer = dataset_factory.get_writer(FLAGS.datasetname)
    writer = dataset_writer()

    # set default graph
    with tf.Graph().as_default():
        # build network
        if FLAGS.use_single_mid:
            real_snippet_len = 1
        else:
            real_snippet_len = FLAGS.snippet_len
        net = networks_factory.build_net(
            FLAGS.netname, n_classes, real_snippet_len,
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
                frames_all = read_n_process(fname_lst)

                # prepare indices
                n_frames = len(lbl_lst)
                left = FLAGS.snippet_len // 2
                right = FLAGS.snippet_len - left

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
                        if FLAGS.use_single_mid:
                            snippet = np.expand_dims(frames_all[frame_id], axis=0)
                        else:
                            snippet = frames_all[frame_id-left:frame_id+right]
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
