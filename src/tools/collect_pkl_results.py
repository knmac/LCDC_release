"""Collect results from multiple pickle results
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

src_pth = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_pth))

import glob
import re
import pickle
import argparse

import numpy as np


def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logdir', type=str,
                        help='Log directory')
    # parser.add_argument('-o', '--output_file', type=str,
    #                     help='file to store output (as csv format)')
    parser.add_argument('-t', '--last', type=int, default=None,
                        help='last N check points to look at')
    args = parser.parse_args()

    assert os.path.exists(args.logdir), '{} not found'.format(args.logdir)
    return args


def natural_key(text):
    """Extract interger key for sorting

    Args:
        text: unformatted text, e.g. 'foo/bar/abc.ckpt-123.pkl'

    Returns:
        key: formatted key, e.g. 123
    """
    key = os.path.basename(text)
    found = re.search('(.ckpt-)(\d+)(.pkl)', key)
    key = int(found.group(2))
    return key


def main():
    """Main function
    """
    # Parse input arguments
    args = parse_args()

    # Retrieve pkl list and sort
    pkl_lst = glob.glob(os.path.join(args.logdir, '*.pkl'))
    assert pkl_lst != [], 'Pickle files not found'
    for item in pkl_lst:
        found = re.search('(.ckpt-)(\d+)(.pkl)', os.path.basename(item))
        if found is None:
            pkl_lst.remove(item)
    pkl_lst.sort(key=natural_key)
    if args.last is not None:
        pkl_lst = pkl_lst[-args.last:]

    # Go through all pkl files
    iter_lst, acc_lst, edit_lst, f1_lst, map_lst = np.array([]), np.array([]),\
        np.array([]), np.array([]), np.array([])
    for fname in pkl_lst:
        data = pickle.load(open(fname, 'rb'))
        iter_lst = np.append(iter_lst, natural_key(fname))
        msg = 'iter {:7} - '.format(natural_key(fname))

        if 'frame_accuracy' in data:
            acc_lst = np.append(acc_lst, data['frame_accuracy'])
            msg += 'acc={:.02f} '.format(data['frame_accuracy'])
        if 'edit' in data:
            edit_lst = np.append(edit_lst, data['edit'])
            msg += 'edit={:.02f} '.format(data['edit'])
        if 'f1' in data:
            f1_lst = np.append(f1_lst, data['f1'])
            msg += 'f1={:.02f} '.format(data['f1'])
        if 'mAP' in data:
            map_lst = np.append(map_lst, data['mAP'])
            msg += 'mAP={:.02f} '.format(data['mAP'])
        print(msg)

    print('--------------------\nOverall results')
    if 'frame_accuracy' in data:
        print('  Accuracy: Mean={:.02f}, Std={:.02f}, Max={:.02f}'.format(
            acc_lst.mean(), acc_lst.std(), acc_lst.max()))
    if 'edit' in data:
        print('  Edit:     Mean={:.02f}, Std={:.02f}, Max={:.02f}'.format(
            edit_lst.mean(), edit_lst.std(), edit_lst.max()))
    if 'f1' in data:
        print('  F1:       Mean={:.02f}, Std={:.02f}, Max={:.02f}'.format(
            f1_lst.mean(), f1_lst.std(), f1_lst.max()))
    if 'mAP' in data:
        print('  mAP:      Mean={:.02f}, Std={:.02f}, Max={:.02f}'.format(
            map_lst.mean(), map_lst.std(), map_lst.max()))
    pass


if __name__ == '__main__':
    main()
