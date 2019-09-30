"""Combine features from appearance and motion streams by concatenation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import scipy.io
import argparse
import numpy as np


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument(
        '--app_dir', type=str,
        help='Directory containing appearance features')
    parser.add_argument(
        '--mot_dir', type=str,
        help='Directory containing motion features')
    parser.add_argument(
        '--out_dir', type=str,
        help='Output directory containing combined features')
    parser.add_argument(
        '--ext', type=str,
        default='mat',
        help='Feature extension')

    args = parser.parse_args()

    assert os.path.isdir(args.app_dir)
    assert os.path.isdir(args.mot_dir)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    return args


def read_tcnfeat(fname):
    """Read TCN features from .mat files. This is the inputs of TCN networks.

    Args:
        fname: path to the .mat feature file

    Returns:
        A: feature, ndarray of shape (N_SAMPLES, DIM)
        Y: groundtruth label, ndarray of shape (N_SAMPLES, 1)
    """
    data = scipy.io.loadmat(fname)
    return data['A'], data['Y']


def combine_feat(app_feat, mot_feat):
    """Combine features from appearance and motion streams by stacking

    Args:
        app_feat: features from appearance stream, ndarray of shape
                  (N_SAMPLES, DIM1)
        mot_feat: features from motion stream, ndarray of shape
                  (N_SAMPLES, DIM2)

    Returns:
        Combined features, ndarray with shape of (N_SAMPLES, DIM1+DIM2)
    """
    assert app_feat.shape[0] == mot_feat.shape[0], 'Different number of samples'
    return np.hstack([app_feat, mot_feat])


def write_comfeat(feat, lbl, fname):
    """Write combined TCN features

    Args:
        feat: combined feature, ndarray of shape (N_SAMPLES, DIM)
        lbl: groundtruth label, ndarray of shape (N_SAMPLES, 1)
        fname: path to the output filename
    """
    mdict = {'A': feat, 'Y': lbl}
    scipy.io.savemat(fname, mdict)

    # foo, bar = read_tcnfeat(fname)
    # assert np.all(foo == feat)
    # assert np.all(bar == lbl)
    pass


def main():
    """Main function"""
    # Retrive lists of features
    app_lst = glob.glob(os.path.join(args.app_dir, '*.'+args.ext))
    app_lst.sort()

    mot_lst = glob.glob(os.path.join(args.mot_dir, '*.'+args.ext))
    mot_lst.sort()

    assert len(app_lst) == len(mot_lst)
    n_files = len(app_lst)

    # Go through each file
    for i in range(n_files):
        # Check name
        assert os.path.basename(app_lst[i]) == os.path.basename(mot_lst[i])

        # Read features
        app_feat, app_lbl = read_tcnfeat(app_lst[i])
        mot_feat, mot_lbl = read_tcnfeat(mot_lst[i])

        # Check label
        assert np.all(app_lbl == mot_lbl)

        # Combine features
        com_feat = combine_feat(app_feat, mot_feat)

        # Save combined features
        fname = os.path.join(args.out_dir, os.path.basename(app_lst[i]))
        write_comfeat(com_feat, app_lbl, fname)
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
