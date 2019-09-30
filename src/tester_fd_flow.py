"""This tester read the flow directly instead of tfrecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import cv2
from progressbar import ProgressBar
from data_utils import dataset_factory
from tester_fd import check_args, eval_on_demand  # The flags from tester_fd are reused
from data_utils import metrics_maker

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'loadfrom',
    None,
    '`flow` or `rgb`. If `flow`, will load npy flow directly. If `rgb`, will '
    'compute flow on the fly. The second one is used for synchronizing with '
    'appearance stream. Should only use with the best checkpoint to save time.')

_DEBUG_ = FLAGS._DEBUG_


def evaluation_sync(net, init_fn, frames, labels, metrics_op, accuracy_vid,
                    summary_op, global_step, confusion_vid, ckpt_fname):
    """Test network with manually created coordinator and logger
    This in in sync with appearance stream

    Returns:
        final_acc_frame: final accuracy on frame level
        final_acc_vid: final accuracy on video level
    """
    n_vids = len(labels)
    snippet_len = FLAGS.snippet_len + 1  # original snippet_len without flow

    with tf.Session() as sess:
        # initialization routine
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        init_fn(sess)

        # check initial metrics
        assert sess.run(confusion_vid).sum() == 0
        assert sess.run(accuracy_vid) == 0

        # summary and checkpoint managers
        sum_writer = tf.summary.FileWriter(FLAGS.testlogdir, sess.graph)

        # go through each test videos
        scores = []
        groundtruths = []
        # Evaluate for the whole snippet
        for vid_id in range(n_vids):
            n_frames = len(labels[vid_id])
            left = snippet_len // 2  # correct, because snippet_len was increased by 1
            right = snippet_len - left

            # go through the video frames in acausal fashion
            frame_id = left
            scores_per_vid = []
            groundtruths_per_vid = []
            while frame_id < n_frames-right+1:
                # produce inputs
                snippet_batch = []
                lbl_batch = []
                for _ in range(FLAGS.batch_size):
                    if frame_id+right > n_frames:
                        break

                    # ignore 1 last frame because this is flow
                    snippet = frames[vid_id][frame_id-left:frame_id+right - 1]
                    # this is in sync with label from appearance stream
                    lbl = labels[vid_id][frame_id]

                    snippet_batch.append(snippet)
                    lbl_batch.append(lbl)
                    frame_id += FLAGS.test_stride
                feed_dict = {net.data_raw: snippet_batch,
                             net.labels_raw: lbl_batch}
                if FLAGS.netname in ['resnet50_vanilla_opticalflow',
                                     'resnet50_vanilla_opticalflow_mid',
                                     'vgg_vanilla_opticalflow']:
                    flow_batch = net.compute_input_flow_gt(snippet_batch)
                    feed_dict = {net.data_raw: snippet_batch,
                                 net.labels_raw: lbl_batch,
                                 net.flow_gt: flow_batch}

                # eval step and update metrics
                (cls_prob,
                 accuracy_vid_,
                 _) = sess.run([net.infer(), accuracy_vid, metrics_op],
                               feed_dict=feed_dict)

                # store results
                scores_per_vid += cls_prob.tolist()
                groundtruths_per_vid += lbl_batch

                # print info frequently
                if frame_id % 100 == 0 and frame_id != 0:
                    tf.logging.info('vid %s step %s: streaming accuracy: %s',
                                    vid_id+1, frame_id+1, accuracy_vid_)

            # concat results
            scores.append(scores_per_vid)
            groundtruths.append(groundtruths_per_vid)

        # make final metrics
        results_dict = metrics_maker.auto_make(scores, groundtruths)
        final_acc = results_dict['frame_accuracy']

        # log final accuracy
        tf.logging.info('Final accuracy: %s', final_acc)

        # add summary
        tf.logging.info('Logging the summary')
        summary_ = sess.run(summary_op)
        sum_writer.add_summary(summary_, sess.run(global_step))
        sum_writer.flush()
        sum_writer.close()

        # log real values of confusion matrices
        if FLAGS.store_conf_mat:
            tf.logging.info('Saving raw confusion matrices as npy file')
            output_fname = os.path.join(FLAGS.testlogdir, ckpt_fname+'.npy')
            confusion_vid_ = sess.run(confusion_vid)
            confusion_dict = {'confusion_vid': confusion_vid_}
            np.save(output_fname, confusion_dict)
    return results_dict


def _load_flows(paths):
    """Load flows and preprocess for testing
    """
    assert isinstance(paths, list)
    # allocate memory
    flows = np.zeros([len(paths), FLAGS.target_height, FLAGS.target_width, 2],
                     dtype=np.float32)

    # load all flows
    pbar = ProgressBar(max_value=len(paths)-1)
    for i in range(len(paths)):
        # Load flow
        flow = np.load(paths[i])

        # Central crop flow
        h, w, _ = flow.shape
        left = (w - FLAGS.target_width) // 2
        top = (h - FLAGS.target_height) // 2
        right = left + FLAGS.target_width
        bottom = top + FLAGS.target_height
        flow = flow[left:right, top:bottom, :]

        # Store flow
        flows[i] = flow
        pbar.update(i)
    return flows


def _compute_flows(paths):
    """Compure optical flow from a list of paths
    """
    assert isinstance(paths, list)

    # allocate memory
    flows = np.zeros([len(paths), FLAGS.target_height, FLAGS.target_width, 2],
                     dtype=np.float32)

    # load all images
    pbar = ProgressBar(max_value=len(paths))
    for i in range(len(paths)-1):
        im1 = skimage.io.imread(paths[i])
        im2 = skimage.io.imread(paths[i+1])

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

        # store images
        flows[i] = flow
        pbar.update(i)

    # Replicate the flow for last frame
    flows[-1] = flow
    return flows


def load_data_from_flow():
    """Load list of frames and corresponding labels.

    Returns:
        frames: a list, each item is a sub-list of all frames per video
        labels: a list, each item is a sub-list of corresponding labels
    """
    # use the load_snippet_pths_test in data writer to get frames and labels
    print('Loading frames and labels...')
    dataset_writer = dataset_factory.get_writer(FLAGS.datasetname)
    writer = dataset_writer()

    # retrieve list of test videos
    vid_lst = writer.generate_data_lst_from_split(FLAGS.split_fn)
    if _DEBUG_:
        vid_lst = vid_lst[:3]

    # for each video, collect fnames and labels with downsampling
    frames, labels = [], []
    print('Found {:d} videos'.format(len(vid_lst)))
    for vid in vid_lst:
        print('  Loading {}...'.format(vid))
        fname_pths_per_vid, labels_per_vid = writer.load_snippet_pths_test(
            FLAGS.datadir, [vid], FLAGS.labels_fname, FLAGS.bg_lbl,
            FLAGS.ext, FLAGS.frameskip)
        fname_pths_per_vid = [x[0] for x in fname_pths_per_vid]

        if _DEBUG_:
            fname_pths_per_vid = fname_pths_per_vid[:200]
            labels_per_vid = labels_per_vid[:200]

        frames.append(_load_flows(fname_pths_per_vid))
        labels.append(np.array(labels_per_vid))
    return frames, labels


def compute_flow_from_images():
    """Load list of frames and corresponding labels.

    Returns:
        frames: a list, each item is a sub-list of all frames per video
        labels: a list, each item is a sub-list of corresponding labels
    """
    # use the load_snippet_pths_test in data writer to get frames and labels
    print('Loading frames and labels...')
    dataset_writer = dataset_factory.get_writer(FLAGS.datasetname)
    writer = dataset_writer()

    # retrieve list of test videos
    vid_lst = writer.generate_data_lst_from_split(FLAGS.split_fn)
    if _DEBUG_:
        vid_lst = vid_lst[:3]

    # for each video, collect fnames and labels with downsampling
    frames, labels = [], []
    print('Found {:d} videos'.format(len(vid_lst)))
    for vid in vid_lst:
        print('  Loading {}...'.format(vid))
        fname_pths_per_vid, labels_per_vid = writer.load_snippet_pths_test(
            FLAGS.datadir, [vid], FLAGS.labels_fname, FLAGS.bg_lbl,
            FLAGS.ext, FLAGS.frameskip)
        fname_pths_per_vid = [x[0] for x in fname_pths_per_vid]

        if _DEBUG_:
            fname_pths_per_vid = fname_pths_per_vid[:200]
            labels_per_vid = labels_per_vid[:200]

        frames.append(_compute_flows(fname_pths_per_vid))
        labels.append(np.array(labels_per_vid))
    return frames, labels


def main(_):
    """Main function"""
    check_args()

    if FLAGS.loadfrom == 'flow':
        frames, labels = load_data_from_flow()
        from tester_fd import evaluation
        eval_fn = evaluation  # orginal eval_fn
    elif FLAGS.loadfrom == 'rgb':
        frames, labels = compute_flow_from_images()
        eval_fn = evaluation_sync
    else:
        print('Unsupported loadfrom')
        sys.exit()

    eval_on_demand(frames, labels, eval_fn)
    pass


if __name__ == '__main__':
    tf.app.run()
