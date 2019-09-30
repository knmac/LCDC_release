"""Extract a single deformable group from the initial offsets and offset at the
center
"""
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import pickle

import numpy as np


def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_pth', type=str,
                        default='./pretrain/Resnet50_deformNet_iter_145000.pkl',
                        help='Path to the pickle file containing saved data')
    parser.add_argument('-g', '--group_index', type=int,
                        help='index of the deformable group')
    args = parser.parse_args()

    assert os.path.exists(args.data_pth), '{} not found'.format(args.data_pth)
    assert args.group_index is not None
    return args


def convert_weights(weights, group_index):
    """Convert weights from (3, 3, ?, 72) to (3, 3, ?, 2)
    """
    # Convert to GKV order
    kh, kw, c, _ = weights.shape
    new_weights = np.reshape(weights, [kh, kw, c, 4, 3, 3, 2])

    # Extract values at the center
    new_weights = new_weights[:, :, :, group_index, 1, 1, :]
    return new_weights


def convert_biases(biases, group_index):
    """Convert biases from (72,) to (2,)
    """
    # Convert to GKV order
    new_biases = np.reshape(biases, [4, 3, 3, 2])

    # Extract values at the center of kernel
    new_biases = new_biases[group_index, 1, 1, :]
    return new_biases


def main():
    """Main function
    """
    # Parse input arguments
    args = parse_args()

    # Load data
    data_dict = pickle.load(open(args.data_pth, 'rb'))

    keys_to_convert = ['res5a_branch2b_offset', 'res5b_branch2b_offset',
                       'res5c_branch2b_offset']
    for item in keys_to_convert:
        data_dict[item]['weights'] = convert_weights(data_dict[item]['weights'], args.group_index)
        data_dict[item]['biases'] = convert_biases(data_dict[item]['biases'], args.group_index)

    # Save new data
    out_pth = args.data_pth.replace('.pkl', '')
    out_pth = '{}_g{}.pkl'.format(out_pth, args.group_index)
    if os.path.exists(out_pth):
        exit()
    with open(out_pth, 'wb') as handle:
        pickle.dump(data_dict, handle)
    pass


if __name__ == '__main__':
    main()
