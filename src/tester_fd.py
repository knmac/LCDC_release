"""This tester read the images directly instead of tfrecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import glob
import numpy as np
import tensorflow as tf
import pickle
import time
import shutil
import skimage.io as sio
from skimage.transform import resize as sresize
from progressbar import ProgressBar

from networks import networks_factory
from data_utils import dataset_factory
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.contrib.framework import get_variables_to_restore
from networks.networks_utils import colorize_tensor
from data_utils import metrics_maker

flags = tf.app.flags
FLAGS = flags.FLAGS

# directories
flags.DEFINE_string('trainlogdir', None, 'where to training logs are stored')
flags.DEFINE_string('testlogdir', None, 'where to test logs are stored')
flags.DEFINE_string('datadir', None, 'where the segmented data are stored'
                                     'e.g. path to the activity folder')

# data reader and processing parameters
flags.DEFINE_string('datasetname', None, 'name of the dataset')
flags.DEFINE_string('split_fn', None, 'filename contain videos ID of a split')
flags.DEFINE_string('labels_fname', None, 'filename containing label description')
flags.DEFINE_string('bg_lbl', 'background', 'name of the background class')
flags.DEFINE_string('ext', 'png', 'extension of frame file names')
flags.DEFINE_boolean('has_bg_lbl', True, 'has background class or not. If'
                                         'True, the number of classes will be'
                                         'increased by 1 from the content of'
                                         '`labels_fname`')

flags.DEFINE_integer('frameskip', None, 'number of frames to skip for downsampling')
flags.DEFINE_integer('batch_size', 1, 'number of snippet per batch')
flags.DEFINE_integer('label_offset', 0, 'labels start with 0 or 1 in dataset')
flags.DEFINE_integer('snippet_len', None, 'number of frames per snippet')
flags.DEFINE_integer('target_height', 224, 'image height after preprocessing')
flags.DEFINE_integer('target_width', 224, 'image height after preprocessing')
flags.DEFINE_boolean('store_conf_mat', False, 'store confusion matrices or not')
flags.DEFINE_integer('test_stride', 1, 'how much frames to stride on testing')

# evaluation parameters
flags.DEFINE_string('netname', None, 'name of the network')
flags.DEFINE_integer('max_time_gap', 1, 'maximum time gap for motion loss')
flags.DEFINE_string('ckpt_fname', None, 'name of the checkpoint'
                    '- `auto`: frequently getting all checkpoints, test the '
                    '          one that has not been tested yet, and backup '
                    '          the best checkpoint'
                    '- `latest`: load the latest checkpoint in trainlogdir'
                    '- `all`: load all checkpoints in trainlogdir'
                    '- (checkpoint name): load the specified checkpoint')

flags.DEFINE_boolean('_DEBUG_', False, 'debug mode')
_DEBUG_ = FLAGS._DEBUG_


def check_args():
    """Check input arguments before running.
    """
    assert os.path.exists(FLAGS.datadir)
    assert os.path.exists(FLAGS.trainlogdir)
    assert os.path.exists(FLAGS.split_fn)
    assert os.path.exists(FLAGS.labels_fname)
    assert FLAGS.snippet_len >= 1
    assert FLAGS.frameskip >= 1
    assert FLAGS.test_stride == 1 or FLAGS.test_stride == FLAGS.snippet_len, \
        'test_stride has to be either 1 or snippet_len (for vanilla+)'
    pass


def evaluation(net, init_fn, frames, labels, metrics_op, accuracy_vid,
               summary_op, global_step, confusion_vid, ckpt_fname):
    """Test network with manually created coordinator and logger

    Returns:
        final_acc_frame: final accuracy on frame level
        final_acc_vid: final accuracy on video level
    """
    n_vids = len(labels)

    with tf.Session() as sess:
        # initialization routine
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        init_fn(sess)

        # check initial metrics
        pyramid_len = 0
        if isinstance(confusion_vid, list):
            assert isinstance(accuracy_vid, list)
            pyramid_len = len(confusion_vid)

        if pyramid_len:
            for item in confusion_vid:
                assert sess.run(item).sum() == 0
            for item in accuracy_vid:
                assert sess.run(item) == 0
        else:
            assert sess.run(confusion_vid).sum() == 0
            assert sess.run(accuracy_vid) == 0

        # summary and checkpoint managers
        sum_writer = tf.summary.FileWriter(FLAGS.testlogdir, sess.graph)

        # go through each test videos
        scores = [[] for _ in range(pyramid_len)]
        groundtruths = []
        # Evaluate for the whole snippet
        if FLAGS.test_stride == 1:
            for vid_id in range(n_vids):
                n_frames = len(labels[vid_id])
                left = FLAGS.snippet_len // 2
                right = FLAGS.snippet_len - left

                # go through the video frames in acausal fashion
                frame_id = left
                scores_per_vid = [[] for _ in range(pyramid_len)]
                groundtruths_per_vid = []
                while frame_id < n_frames-right+1:
                    # produce inputs
                    snippet_batch = []
                    lbl_batch = []
                    for _ in range(FLAGS.batch_size):
                        if frame_id+right > n_frames:
                            break
                        snippet = frames[vid_id][frame_id-left:frame_id+right]
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
                    if pyramid_len:
                        for scale in range(pyramid_len):
                            scores_per_vid[scale] += cls_prob[scale].tolist()
                    else:
                        scores_per_vid += cls_prob.tolist()
                    groundtruths_per_vid += lbl_batch

                    # print info frequently
                    if frame_id % 100 == 0 and frame_id != 0:
                        tf.logging.info('vid %s step %s: streaming accuracy: %s',
                                        vid_id+1, frame_id+1, accuracy_vid_)

                # concat results
                if pyramid_len:
                    for scale in range(pyramid_len):
                        scores[scale].append(scores_per_vid[scale])
                else:
                    scores.append(scores_per_vid)
                groundtruths.append(groundtruths_per_vid)

            # make final metrics
            if pyramid_len:
                results_dict = []
                final_acc = []
                for scale in range(pyramid_len):
                    tf.logging.info('Scale # ' + str(scale+1))
                    results_dict.append(metrics_maker.auto_make(scores[scale],
                                                                groundtruths))
                    final_acc.append(results_dict[scale]['frame_accuracy'])
            else:
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
            pass

        # Evaluate for each frame separately, used for vanilla+
        elif FLAGS.test_stride == FLAGS.snippet_len and FLAGS.snippet_len != 1:
            for vid_id in range(n_vids):
                n_frames = len(labels[vid_id])
                left = FLAGS.snippet_len // 2
                right = FLAGS.snippet_len - left

                # go through the video frames in acausal fashion
                frame_id = left
                scores_per_vid = [[] for _ in range(pyramid_len)]
                groundtruths_per_vid = []
                while frame_id < n_frames-right+1:
                    # produce inputs
                    snippet_batch = []
                    lbl_batch = []
                    for _ in range(FLAGS.batch_size):
                        if frame_id+right > n_frames:
                            break
                        snippet = frames[vid_id][frame_id-left:frame_id+right]
                        lbl = labels[vid_id][frame_id-left:frame_id+right]
                        snippet_batch.append(snippet)
                        lbl_batch.append(lbl)
                        frame_id += FLAGS.test_stride
                    lbl_batch = np.concatenate(lbl_batch).tolist()
                    feed_dict = {net.data_raw: snippet_batch}

                    # eval step and update metrics
                    cls_prob = sess.run(net.infer(), feed_dict=feed_dict)

                    # store results
                    if pyramid_len:
                        for scale in range(pyramid_len):
                            scores_per_vid[scale] += cls_prob[scale].tolist()
                    else:
                        scores_per_vid += cls_prob.tolist()
                    groundtruths_per_vid += lbl_batch

                # concat results
                if pyramid_len:
                    for scale in range(pyramid_len):
                        scores[scale].append(scores_per_vid[scale])
                else:
                    scores.append(scores_per_vid)
                groundtruths.append(groundtruths_per_vid)

            # make final metrics
            if pyramid_len:
                results_dict = []
                final_acc = []
                for scale in range(pyramid_len):
                    results_dict.append(metrics_maker.auto_make(scores[scale],
                                                                groundtruths))
                    final_acc.append(results_dict[scale]['frame_accuracy'])
            else:
                results_dict = metrics_maker.auto_make(scores, groundtruths)
                final_acc = results_dict['frame_accuracy']

            # log final accuracy
            tf.logging.info('Final accuracy: %s', final_acc)
            pass
    return results_dict


def eval_one(ckpt_fname, frames, labels, eval_fn):
    """ Run evaluation on a specific checkpoint

    Args:
        ckpt_fname: file name of the checkpoint to restore from
        frames: a list, each item is a sub-list of all frames per video
        labels: a list, each item is a sub-list of corresponding labels

    Returns:
        final_acc_frame: final accuracy on frame level
        final_acc_vid: final accuracy on video level
    """
    # retrieve label dictionary
    lbl_list = open(FLAGS.labels_fname).read().splitlines()
    n_classes = len(lbl_list)
    if FLAGS.has_bg_lbl:
        n_classes += 1

    # set verbosity for info level only
    tf.logging.set_verbosity(tf.logging.INFO)

    # contruct graph and build models
    with tf.Graph().as_default():
        tf.logging.info('Evaluating checkpoint %s', ckpt_fname)

        # build network
        net = networks_factory.build_net(
            FLAGS.netname, n_classes, FLAGS.snippet_len,
            FLAGS.target_height, FLAGS.target_width,
            max_time_gap=FLAGS.max_time_gap,
            trainable=False)

        # restore checkpoint function
        global_step = get_or_create_global_step()
        variables_to_restore = get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def init_fn(sess):
            tf.logging.info('Restoring checkpoint...')
            return saver.restore(sess, ckpt_fname)

        # metrics to predict, has to be defined after restoring checkpoint
        # unlike in training, otherwise it will reload the old values
        (accuracy_vid, confusion_vid, metrics_op) = net.create_metrics()
        if isinstance(confusion_vid, list):
            confusion_vid_img = [colorize_tensor(x, extend=True) for x in confusion_vid]
        else:
            confusion_vid_img = colorize_tensor(confusion_vid, extend=True)

        # summaries
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
        summary_op = tf.summary.merge_all()

        # evaluation phase
        results_dict = eval_fn(net, init_fn, frames, labels, metrics_op,
                               accuracy_vid, summary_op, global_step,
                               confusion_vid, os.path.basename(ckpt_fname))
    return results_dict


def eval_multiple(ckpt_fnames, frames, labels, eval_fn, averaging=False):
    """ Run evaluation once on a specific checkpoint

    Args:
        ckpt_fnames: a list of checkpoints to restore from
        frames: a list, each item is a sub-list of all frames per video
        labels: a list, each item is a sub-list of corresponding labels
        averaging: whether to show the average results
    """
    mean_acc, mean_map = 0.0, 0.0
    pyramid_len = 0

    for ckpt_fname in ckpt_fnames:
        if not tf.train.checkpoint_exists(ckpt_fname):
            print('Checkpoint does not exist', ckpt_fname)
            continue
        results_dict = eval_one(ckpt_fname, frames, labels, eval_fn)
        if isinstance(results_dict, list):
            if not isinstance(mean_acc, list):
                mean_acc = [mean_acc]
                mean_map = [mean_map]
                pyramid_len = len(results_dict)

        output_fname = os.path.join(
            FLAGS.testlogdir,
            'results_dict_{}.pkl'.format(os.path.basename(ckpt_fname)))
        with open(output_fname, 'wb') as f:
            pickle.dump(results_dict, f)

        if pyramid_len:
            for scale in range(pyramid_len):
                mean_acc[scale] += results_dict[scale]['frame_accuracy']
                mean_map[scale] += results_dict[scale]['mAP']
        else:
            mean_acc += results_dict['frame_accuracy']
            mean_map += results_dict['mAP']

    if averaging:
        print('Overall acc:', np.array(mean_acc) / len(ckpt_fnames))
        print('Overall mAP:', np.array(mean_map) / len(ckpt_fnames))
    pass


def backup_ckpt(ckpt_fname, dest_dir, results_dict, remove_old=True):
    """Back up checkpoint files

    Args:
        ckpt_fname: name of the checkpoint. Does not include suffixes, e.g.
            `index`, `data`, or `meta`
        dest_dir: destination directories to backup checkpoint
        accuracy: accuracy value of the current checkpoint
        remove_old: whether to remove the old checkpoint or not
    """
    # remove old checkpoints
    if os.path.exists(dest_dir) and remove_old:
        shutil.rmtree(dest_dir)

    # make directory if not exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # copy necessary files
    file_lst = glob.glob(os.path.join(ckpt_fname + '.*'))
    for fname in file_lst:
        shutil.copy(fname, dest_dir)

    # record results
    with open(os.path.join(dest_dir, 'acc.txt'), 'w') as f:
        if isinstance(results_dict, list):
            acc = []
            for scale in results_dict:
                acc.append(scale['frame_accuracy'])
        else:
            acc = results_dict['frame_accuracy']
        f.write('{}\n'.format(acc))
    with open(os.path.join(dest_dir, 'results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)
    pass


def _load_images(paths):
    """Load images and preprocess for testing

    Returns:
        images: list of images with resizing and mean removal
    """
    assert isinstance(paths, list)
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    # allocate memory
    images = np.zeros([len(paths), FLAGS.target_height, FLAGS.target_width, 3],
                      dtype=np.float32)

    # load all images
    pbar = ProgressBar(max_value=len(paths))
    for i in range(len(paths)):
        img = sio.imread(paths[i])

        # resize images
        img = sresize(img, (FLAGS.target_height, FLAGS.target_width, 3),
                      mode='constant', preserve_range=True)

        # store images
        images[i] = img.astype(np.float32)
        pbar.update(i)

    # mean removal
    images -= [_R_MEAN, _G_MEAN, _B_MEAN]
    return images


def load_data():
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

        frames.append(_load_images(fname_pths_per_vid))
        labels.append(np.array(labels_per_vid))
    return frames, labels


def get_checkpoint_list(dir):
    """Retrieve sorted list of checkpoints

    Args:
        dir: path to the directory containing checkpoints

    Return:
        ckpt_fnames: list of checkpoints, sorted by natural order
    """
    ckpt_fnames = glob.glob(os.path.join(dir, '*.index'))
    ckpt_fnames = [x.replace('.index', '') for x in ckpt_fnames]
    ckpt_fnames.sort(key=lambda key: int(os.path.basename(key).split('-')[-1]))
    return ckpt_fnames


def eval_on_demand(frames, labels, eval_fn):
    # retrieve the checkpoints
    if FLAGS.ckpt_fname == 'auto':
        best_dir = os.path.join(FLAGS.testlogdir, 'best')

        # load the visited list
        visited_fname = os.path.join(FLAGS.testlogdir, 'visited.pkl')
        if os.path.exists(visited_fname):
            visited_dict = pickle.load(open(visited_fname, 'rb'))
            visited_ckpt = visited_dict['ckpt']
            visited_final_acc = visited_dict['final_acc']
        else:
            visited_ckpt = []
            visited_final_acc = []

        # keep looping
        while True:
            # get all possible checkpoints
            ckpt_fnames = get_checkpoint_list(FLAGS.trainlogdir)
            # ckpt_state = tf.train.get_checkpoint_state(FLAGS.trainlogdir)
            # ckpt_fnames = ckpt_state.all_model_checkpoint_paths

            # go through each checkpoint
            for ckpt_fname in ckpt_fnames:
                # ignore non-existing checkpoints
                ckpt_fname = os.path.join(FLAGS.trainlogdir,
                                          os.path.basename(ckpt_fname))
                if not tf.train.checkpoint_exists(ckpt_fname):
                    print('Checkpoint does not exist', ckpt_fname)
                    continue

                # ignore visited checkpoints
                if ckpt_fname in visited_ckpt:
                    continue

                # evaluate the checkpoint
                results_dict = eval_one(ckpt_fname, frames, labels, eval_fn)
                pyramid_len = 0
                if isinstance(results_dict, list):
                    final_acc = [scale['frame_accuracy'] for scale in results_dict]
                    pyramid_len = len(results_dict)
                else:
                    final_acc = results_dict['frame_accuracy']

                # update the visited list
                visited_ckpt.append(ckpt_fname)
                if pyramid_len:
                    visited_final_acc.append(max(final_acc))
                else:
                    visited_final_acc.append(final_acc)

                # backup the results of that ckpt
                output_fname = os.path.join(
                    FLAGS.testlogdir,
                    'results_dict_{}.pkl'.format(os.path.basename(ckpt_fname)))
                with open(output_fname, 'wb') as f:
                    pickle.dump(results_dict, f)

                # backup the best checkpoint
                if pyramid_len:
                    tmp = max(final_acc)
                else:
                    tmp = final_acc
                if tmp >= max(visited_final_acc):
                    backup_ckpt(ckpt_fname, best_dir, results_dict)

                # save the visited list
                visited_dict = {'ckpt': visited_ckpt,
                                'final_acc': visited_final_acc}
                pickle.dump(visited_dict, open(visited_fname, 'wb'))

            # wait for some time before running the next batch of job
            print('Finished testing. Sleeping and wait for new checkpoints...')
            sleep_time_in_min = 10.0
            time.sleep(sleep_time_in_min * 60)
        pass  # end of auto ckpt_fname
    elif FLAGS.ckpt_fname == 'latest':
        print('Loading the latest checkpoint in', FLAGS.trainlogdir)
        ckpt_fname = tf.train.latest_checkpoint(FLAGS.trainlogdir)
        results_dict = eval_one(ckpt_fname, frames, labels, eval_fn)
        with open(os.path.join(FLAGS.testlogdir, 'results_dict.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)
    elif FLAGS.ckpt_fname == 'all':
        print('Loading all checkpoints in', FLAGS.trainlogdir)
        ckpt_fnames = get_checkpoint_list(FLAGS.trainlogdir)
        # ckpt_state = tf.train.get_checkpoint_state(FLAGS.trainlogdir)
        # ckpt_fnames = ckpt_state.all_model_checkpoint_paths
        eval_multiple(ckpt_fnames, frames, labels, eval_fn)
    elif 'last_' in FLAGS.ckpt_fname:
        try:
            n_ckpts = int(FLAGS.ckpt_fname.replace('last_', ''))
        except Exception:
            print('Wrong format `last_[n_checkpoints]`. Received ',
                  FLAGS.ckpt_fname)
            sys.exit()
        print('Loading {} last checkpoints in {}'.format(n_ckpts, FLAGS.trainlogdir))
        ckpt_fnames = get_checkpoint_list(FLAGS.trainlogdir)
        # ckpt_state = tf.train.get_checkpoint_state(FLAGS.trainlogdir)
        # ckpt_fnames = ckpt_state.all_model_checkpoint_paths[-n_ckpts:]
        eval_multiple(ckpt_fnames, frames, labels, eval_fn, averageing=True)
    else:
        assert tf.train.checkpoint_exists(FLAGS.ckpt_fname), \
            'Checkpoint does not exist'
        ckpt_fname = FLAGS.ckpt_fname
        results_dict = eval_one(ckpt_fname, frames, labels, eval_fn)
        with open(os.path.join(FLAGS.testlogdir, 'results_dict.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)
    pass


def main(_):
    """Main function"""
    check_args()
    frames, labels = load_data()
    eval_fn = evaluation  # function pointer
    eval_on_demand(frames, labels, eval_fn)
    pass


if __name__ == '__main__':
    tf.app.run()
