"""Visualize segmentation results
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import sys
import os

src_pth = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_pth))

from data_utils.metrics_maker import frame_accuracy as frame_accuracy

import argparse
import pickle

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--lcdc_results', type=str,
        help='File containing results')
    parser.add_argument(
        '--tcn_results', type=str,
        help='File containing results')
    parser.add_argument(
        '--split', type=str,
        help='Test split filename')

    args = parser.parse_args()

    assert os.path.isfile(args.lcdc_results)
    assert os.path.isfile(args.tcn_results)
    assert os.path.isfile(args.split)
    return args


def load_data(fname, protocol):
    """Load data from filename
    """
    if protocol == 'LCDC-output':
        data = pickle.load(open(fname, 'rb'))
        predictions = data['y_pred']
        scores = data['y_score_in']
        groundtruths = data['y_true_in']

        scores = [np.array(sc) for sc in scores]
        groundtruths = [np.array(gt) for gt in groundtruths]

        # NOTE: experimental
        # step = 6  # 50salads
        # step = 5  # gtea
        # print('Saved acc:       {}'.format(data['frame_accuracy']))
        # print('Recomputed acc:  {}'.format(frame_accuracy(predictions, groundtruths)))
        # foo = [x[::step] for x in predictions]
        # bar = [x[::step] for x in groundtruths]
        # print('Downsampled acc: {}'.format(frame_accuracy(foo, bar)))
    elif protocol == 'TCN-output':
        data = scipy.io.loadmat(fname)
        predictions = data['P'].squeeze()
        scores = data['S'].squeeze()
        groundtruths = data['Y'].squeeze()

        predictions = [pr.squeeze() for pr in predictions]
        scores = scores.tolist()
        groundtruths = [gt.squeeze() for gt in groundtruths]

    assert len(predictions) == len(scores) and len(scores) == len(groundtruths)
    return predictions, scores, groundtruths


def viz(fname, splits, protocol, figname):
    """Visualize
    """
    # Load data
    predictions, scores, groundtruths = load_data(fname, protocol)
    assert len(splits) == len(predictions)

    # Visualize
    n_vids = len(splits)
    n_classes = scores[0].shape[1]
    fig, axes = plt.subplots(n_vids, 1, figsize=(10, 8))
    for i in range(n_vids):
        pred = predictions[i] / float(n_classes - 1)
        gt = groundtruths[i] / float(n_classes - 1)
        data2draw = np.vstack([pred, gt])

        axes[i].imshow(
            data2draw, interpolation='nearest', vmin=0, vmax=1)
        axes[i].set_aspect('auto')
        axes[i].set_xticks([])
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(['predictions', 'groundtruths'])
        axes[i].set_ylabel(splits[i], rotation='horizontal')
        axes[i].yaxis.set_label_position('right')

    fig.suptitle(figname)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return 0


def main():
    """Main function
    """
    splits = open(args.split, 'r').read().splitlines()
    splits = [x for x in splits if x != '']

    figname = '{} - pre-EDTCN'.format(args.split.split('/')[-2])
    viz(args.lcdc_results, splits, 'LCDC-output', figname)

    figname = '{} - post-EDTCN'.format(args.split.split('/')[-2])
    viz(args.tcn_results, splits, 'TCN-output', figname)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
