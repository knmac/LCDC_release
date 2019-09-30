import os
import glob
import argparse
from collections import OrderedDict
import numpy as np


def parse_args():
    """ Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str,
        default='./data/50_salads_dataset')
    parser.add_argument(
        '-s', '--segmented_activities', type=str,
        default='activity')
    parser.add_argument(
        '-l', '--labels', type=str,
        default='labels')
    parser.add_argument(
        '-v', '--level', type=str,
        default='low')

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    assert args.level == 'low' or args.level == 'mid'
    return args


def process_key(activity, level):
    """ Return the actual activity class name, according to the given level
    """
    key = activity[activity.index('_')+1:]
    if level == 'mid':
        key = key[:key.rindex('_')]
    return key


def main():
    """Main function"""
    args = parse_args()

    # retrieve classes
    lbl_pth = os.path.join(
        args.dataset, args.labels, 'actions_{}lvl.txt'.format(args.level))
    assert os.path.exists(lbl_pth)
    classes = open(lbl_pth).read().splitlines()

    # retrieve list of video paths
    segmented_pth = os.path.join(
        args.dataset, args.segmented_activities)
    assert os.path.exists(segmented_pth)
    vid_list = glob.glob(segmented_pth + '/*')

    # prepare frequency dictionary
    freq = OrderedDict()
    for cls in classes:
        freq[cls] = []

    # go through each segmented videos
    for vid_pth in vid_list:
        activities = os.listdir(vid_pth)
        activities.sort()
        for activity in activities:
            # ignore keys not in the analyzing level
            key = process_key(activity, args.level)
            if key not in classes:
                continue

            # count number of frames of this activity
            n_frames = len(glob.glob(os.path.join(
                vid_pth, activity, '*.jpg')))

            # append to dictionary
            freq[key].append(n_frames)

    # analyze
    print('Number of frames per activity class')
    for key in freq.keys():
        foo = freq[key]
        print('    %s\tmin=%d\tmax=%d\tmedian=%d' % \
                (key, np.min(foo), np.max(foo), np.median(foo)))
    print(80*'-' + '\n')
    pass


if __name__ == '__main__':
    main()
