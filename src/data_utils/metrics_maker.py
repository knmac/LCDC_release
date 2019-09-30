"""Create multiple metrics
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '..', '..', 'src', 'data_utils')))

import numpy as np

from sklearn.metrics import confusion_matrix
from scipy import signal
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import itertools

from lea_metrics import accuracy, edit_score, overlap_f1, mid_mAP


def conf_mat(y_pred, y_true, normalize=False):
    """Find confusion matrix for frame-wise accuracy

    Args:
        y_pred: list of prediction vectors
        y_true: list of groundtruth vectors

    Returns:
        frame-wise confusion matrix
    """
    y_pred_all = np.concatenate(y_pred)
    y_true_all = np.concatenate(y_true)

    cm = confusion_matrix(y_true_all, y_pred_all)

    if normalize:
        cmsum = cm.sum(axis=1)[:, np.newaxis] + 1e-10
        cm = cm.astype('float') / cmsum
    return cm


def auto_make(y_score_in, y_true_in, downsampling=1, median_size=5):
    """Automatically make result dictionary

    Args:
        y_score_in: input score data, list of list of list following the order
            of: videos, samples, classes
        y_score_in: input groundtruth, list of list following the order of:
            videos, samples
        downsampling: downsampling rate, default=1 means no downsampling
        median_size: size of median filter. This is just for reference

    Returns:
        results_dict: dictionary of results
    """
    print('Computing metrics...')
    n_vids = len(y_true_in)
    n_classes = len(y_score_in[0][0])
    bg_class = 0

    # downsample if needed
    y_score = []
    y_true = []
    y_pred = []
    y_pred_median = []
    for i in range(n_vids):
        a_score = np.copy(y_score_in[i])[::downsampling, :]
        y_score.append(np.array(a_score))
        y_pred.append(np.argmax(a_score, axis=1))
        y_true.append(np.copy(y_true_in[i])[::downsampling])

    # post-processing
    y_pred_median = []
    for i in range(n_vids):
        filtered = signal.medfilt(np.copy(y_pred[i]), median_size)
        y_pred_median.append(filtered.astype(np.int))

    # compute all metrics------------------------------------------------------
    results_dict = {}
    results_dict['y_score_in'] = y_score_in
    results_dict['y_true_in'] = y_true_in
    results_dict['downsampling'] = downsampling
    results_dict['y_pred'] = y_pred
    # results_dict['y_pred_median'] = y_pred_median
    results_dict['median_size'] = median_size
    results_dict['conf_mat'] = conf_mat(y_pred, y_true)

    acc = accuracy(y_pred, y_true)
    edit = edit_score(y_pred, y_true, True, bg_class)
    f1 = overlap_f1(y_pred, y_true, n_classes, bg_class)
    precisions, mAP = mid_mAP(y_pred, y_true, y_score, bg_class)

    results_dict['frame_accuracy'] = acc
    results_dict['edit'] = edit
    results_dict['f1'] = f1
    results_dict['precisions'] = precisions
    results_dict['mAP'] = mAP

    # print results------------------------------------------------------------
    print('>' * 80)
    print('Frame-wise accuracy: {:.02f}'.format(acc))
    print('Edit: {:.02f}'.format(edit))
    print('Overlap F1: {:.02f}'.format(f1))
    # print('Midpoint-hit criterion metrics')
    # print('  precisions: ', results_dict['precisions'])
    print('mAP: {:.02f}'.format(mAP))
    print('<' * 80)
    return results_dict


def _viz_pred(y_pred, y_true):
    """Visualize prediction results

    Args:
        y_pred: list of prediction vectors
        y_true: list of groundtruth vectors
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, 0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, 0)

    plt.figure()

    # detection results
    plt.subplot(211)
    plt.imshow(y_pred, aspect='auto', interpolation='nearest')
    plt.yticks([])
    plt.ylabel('detection')
    plt.tight_layout()
    plt.colorbar()

    # groundtruth
    plt.subplot(212)
    plt.imshow(y_true, aspect='auto', interpolation='nearest')
    plt.yticks([])
    plt.ylabel('groundtruth')
    plt.tight_layout()
    plt.colorbar()
    pass


def _viz_confmat(cm, label_dict, normalize=True):
    """Visualize confusion matrix

    Args:
        cm: confusion matrix
        label_dict: list of labels (with background). If None, will use numbers
    """
    if normalize:
        cmsum = cm.sum(axis=1)[:, np.newaxis] + 1e-10
        cm = cm.astype('float') / cmsum

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    if label_dict is not None:
        tick_marks = np.arange(len(label_dict))
        plt.xticks(tick_marks, label_dict, rotation=45)
        plt.yticks(tick_marks, label_dict)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    pass


def reload_and_visualize(results_dict, label_dict, video_index,
                         new_downsampling, new_median_size,
                         visualize):
    """Load and visualize saved results with new_downsampling and
    new_median_size if needed

    Args:
        results_dict: dictionary of results
        label_dict: list of all labels (with background)
        video_index: index of video to visualize detection results
        new_downsampling: new downsampling rate
        new_median_size: new median size
        visualize: visualize the results
    """
    # downsampling again if necessary
    results_dict = auto_make(results_dict['y_score_in'],
                             results_dict['y_true_in'],
                             new_downsampling, new_median_size)

    if visualize:
        # visualize one prediction results
        _viz_pred(results_dict['y_pred'][video_index],
                  results_dict['y_true_in'][video_index])

        # visualize confusion matrix
        _viz_confmat(results_dict['conf_mat'], label_dict)
        plt.show()
    pass


if __name__ == '__main__':
    """Interface to call from command line
    """
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dict', type=str,
                        help='path to results_dict')
    parser.add_argument('-l', '--label_dict', type=str,
                        help='path to label_dict. If ignored, it will use '
                             'numbers instead')
    parser.add_argument('-i', '--video_index', type=int, default=0,
                        help='index of video to visualize detection')
    parser.add_argument('-d', '--new_downsampling', type=int, default=1,
                        help='new downsampling. If ignored, there will be no '
                             'downsampling')
    parser.add_argument('-m', '--new_median_size', type=int, default=5,
                        help='new size for median filter (only for reference '
                             'purpose)')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help='1 if visualize, 0 otherwise')
    args = parser.parse_args()
    assert os.path.exists(args.results_dict)

    # load data
    results_dict = pickle.load(open(args.results_dict, 'rb'))
    if args.label_dict is not None and os.path.exists(args.label_dict):
        label_dict = open(args.label_dict).read().splitlines()
        label_dict = ['background'] + label_dict
    else:
        label_dict = None

    # reload and visualize
    reload_and_visualize(results_dict, label_dict, args.video_index,
                         args.new_downsampling, args.new_median_size,
                         args.visualize)
