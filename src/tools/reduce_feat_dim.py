"""Reduce feature dimension with PCA
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import glob
import argparse
import scipy.io as sio
import numpy as np

from sklearn.decomposition import PCA


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir_in', type=str,
        help='input directory')
    parser.add_argument(
        '--dir_out', type=str,
        help='output directory')
    parser.add_argument(
        '--ext', type=str, default='.avi.mat',
        help='feature extension')
    parser.add_argument(
        '--train_split', type=str,
        help='training split')
    parser.add_argument(
        '--test_split', type=str,
        help='testing split')
    parser.add_argument(
        '--pca_dim', type=int, default=1024,
        help='new feature dimension after reduction')
    parser.add_argument(
        '--dataset', type=str, choices=['50salads', 'gtea'],
        help='dataset name')

    args = parser.parse_args()
    assert os.path.isdir(args.dir_in)
    assert os.path.isfile(args.train_split)
    assert os.path.isfile(args.test_split)

    if not os.path.isdir(args.dir_out):
        os.makedirs(args.dir_out)
    return args


def load_train_test(dir_in, train_split_pth, test_split_pth, ext, dataset):
    """Load training and testing fname lists
    """
    fnames = glob.glob(os.path.join(dir_in, '*'+ext))
    fnames.sort()

    train_split = open(train_split_pth).read().splitlines()
    test_split = open(test_split_pth).read().splitlines()

    train_fnames, test_fnames = [], []
    for fname in fnames:
        tmp = os.path.basename(fname)
        tmp = tmp.replace(ext, '')
        if dataset == '50salads':
            tmp = tmp.replace('rgb-', '')
        elif dataset == 'gtea':
            tmp = tmp

        assert (tmp in train_split or tmp in test_split)
        if tmp in train_split:
            train_fnames.append(fname)
        elif tmp in test_split:
            test_fnames.append(fname)
    return train_fnames, test_fnames


def reduce_dim(pca, fnames, dir_out):
    """Reduce dimension
    """
    for fname in fnames:
        data = sio.loadmat(fname)
        feat_in = data['A']
        lbl = data['Y']
        feat_out = pca.transform(feat_in)

        fname_out = os.path.join(dir_out, os.path.basename(fname))
        mdict = {'A': feat_out, 'Y': lbl}
        sio.savemat(fname_out, mdict)


def main():
    """Main function"""
    # Retrive file names
    train_fnames, test_fnames = load_train_test(
        args.dir_in, args.train_split, args.test_split, args.ext, args.dataset)

    # Read input features
    all_train_feat = []
    # all_train_lbl = []
    for fname in train_fnames:
        data = sio.loadmat(fname)
        all_train_feat.append(data['A'])
        # all_train_lbl.append(data['Y'])

    # Learn pca
    pca = PCA(n_components=args.pca_dim)
    pca.fit(np.vstack(all_train_feat))

    # Reduce dimension
    reduce_dim(pca, train_fnames, args.dir_out)
    reduce_dim(pca, test_fnames, args.dir_out)
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
