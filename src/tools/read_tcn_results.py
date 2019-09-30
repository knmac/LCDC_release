"""Wrapper to read TCN mat file results
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

src_pth = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_pth))

import numpy as np
import scipy
import glob
import argparse
from data_utils.lea_metrics import accuracy, edit_score, overlap_f1, mid_mAP


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--result_dir', type=str,
                        help='Path to TCN result directory (with many runs)')
    parser.add_argument('-e', '--ext', type=str, default='.mat',
                        help='file extension')
    args = parser.parse_args()

    assert os.path.isdir(args.result_dir)
    return args


def main():
    """Main function"""
    # Load run list
    run_lst = os.listdir(args.result_dir)
    run_lst.sort()

    all_acc = []
    all_edit = []
    all_f1 = []
    all_map = []
    for run in run_lst:
        split_lst = glob.glob(os.path.join(args.result_dir, run, '*'+args.ext))
        split_lst.sort()
        for split_pth in split_lst:
            # Load data
            data = scipy.io.loadmat(split_pth)
            P, S, Y = data['P'].squeeze(), data['S'].squeeze(), data['Y'].squeeze()
            P = [x.squeeze() for x in P]
            S = S.tolist()
            Y = [x.squeeze() for x in Y]

            # Compute metrics
            acc = accuracy(P, Y)
            edit = edit_score(P, Y, norm=True, bg_class=0)
            f1 = overlap_f1(P, Y, n_classes=18, bg_class=0)
            _, mAP = mid_mAP(P, Y, S, bg_class=0)

            all_acc.append(acc)
            all_edit.append(edit)
            all_f1.append(f1)
            all_map.append(mAP)

    all_acc = np.array(all_acc)
    all_edit = np.array(all_edit)
    all_f1 = np.array(all_f1)
    all_map = np.array(all_map)
    print('Acc:   mean={:.02f}, std={:.02f}'.format(all_acc.mean(), all_acc.std()))
    print('Edit:  mean={:.02f}, std={:.02f}'.format(all_edit.mean(), all_edit.std()))
    print('F1@10: mean={:.02f}, std={:.02f}'.format(all_f1.mean(), all_f1.std()))
    print('mAP:   mean={:.02f}, std={:.02f}'.format(all_map.mean(), all_map.std()))
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
