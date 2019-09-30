"""Naive linear fusion for baseline with appearance stream ResNet and motion
stream VGG16
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import argparse
import pickle
import numpy as np
from data_utils import metrics_maker

_SEARCH_WEIGHT = False  # True if search the best combination weight


def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--motion_pth', type=str,
                        help='path to motion stream results dict')
    parser.add_argument('-a', '--appear_pth', type=str,
                        help='path to appearance stream results dict')
    parser.add_argument('-d', '--downsample_rate', type=int, default=1,
                        help='path to appearance stream results dict')
    args = parser.parse_args()
    assert args.appear_pth is not None and os.path.isfile(args.appear_pth)
    assert args.motion_pth is not None and os.path.isfile(args.motion_pth)
    return args


def linear_combine(appear_w, auto_make=False):
    """Linearly combine the scores

    Args:
        appear_w: weight for appearance stream. Weight for motion stream is
            computed as (1 - appear_w)

    Return:
        acc: framewise accuracy
    """
    appear_results = pickle.load(open(args.appear_pth, 'rb'))
    motion_results = pickle.load(open(args.motion_pth, 'rb'))
    n_vids = len(appear_results['y_true_in'])

    # check groundtruth
    for i in range(n_vids):
        assert appear_results['y_true_in'][i] == motion_results['y_true_in'][i]

    # combine score
    score_appear = appear_results['y_score_in']
    score_motion = motion_results['y_score_in']
    gt = motion_results['y_true_in']

    if args.downsample_rate != 1:
        score_appear = [x[::args.downsample_rate] for x in score_appear]
        score_motion = [x[::args.downsample_rate] for x in score_motion]
        gt = [x[::args.downsample_rate] for x in gt]

    score_combine, pred_combine = [], []
    for i in range(n_vids):
        foo = np.array(score_appear[i])
        bar = np.array(score_motion[i])
        tmp = appear_w*foo + (1-appear_w)*bar
        score_combine.append(tmp)
        pred_combine.append(tmp.argmax(axis=1))

    if auto_make:
        print('Appearance stream:')
        metrics_maker.auto_make(score_appear, gt)
        print('\n')

        print('Motion stream:')
        metrics_maker.auto_make(score_motion, gt)
        print('\n')

        print('Two streams:')
        metrics_maker.auto_make(score_combine, gt)
    acc = metrics_maker.accuracy(pred_combine, gt)
    return acc


def main():
    """Main function"""
    if _SEARCH_WEIGHT:
        maxacc = 0.0
        bestw = 0
        for appear_w in np.arange(0.5, 1.0, 0.005):
            output = linear_combine(appear_w)
            if output > maxacc:
                maxacc = output
                bestw = appear_w
    else:
        bestw = 0.5

    print('Appearance stream only: {:.02f}'.format(linear_combine(1.0)))
    print('Motion stream only:     {:.02f}'.format(linear_combine(0.0)))
    print('Two-stream:             {:.02f} (appear_w={:.03f})'.format(
        linear_combine(bestw), bestw))

    linear_combine(bestw, auto_make=True)
    pass


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
