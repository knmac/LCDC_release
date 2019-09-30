"""Segment activities for GTEA dataset
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import glob
import argparse
import shutil
from progressbar import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framedir', type=str,
                        help='Location of extracted frames')
    parser.add_argument('--labeldir', type=str,
                        help='Directory storing labels as mat files')
    parser.add_argument('--outputdir', type=str,
                        help='Output directory')
    parser.add_argument('--labeldict_pth', type=str,
                        help='Label dictionary path')
    parser.add_argument('--ext', type=str,
                        help='image extension')
    parser.add_argument('--bg_lbl', type=str,
                        help='label for background class')

    args = parser.parse_args()
    assert os.path.exists(args.framedir)
    assert os.path.exists(args.labeldir)
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    return args


def load_label_dict(labeldict_pth):
    """Load label dictionary

    Args:
        labeldict_pth: path to the label description

    Return:
        lbl_dict: label dictionary {lbl_name: lbl_id}
        inv_lbl_dict: inverse label dictionary {lbl_id, lbl_name}
    """
    lbl_list = open(labeldict_pth).read().splitlines()
    lbl_dict = {}
    inv_lbl_dict = {}
    for i in range(len(lbl_list)):
        lbl_dict[lbl_list[i]] = i
        inv_lbl_dict[i] = lbl_list[i]
    return lbl_dict, inv_lbl_dict


def parse_label(label_fname, inv_lbl_dict):
    """Parse label information from mat files

    Args:
        label_fname: full path of the mat file, storing label information
        inv_lbl_dict: inverse label dictionary

    Return:
        start: list of start frame id
        stop: list of stop frame id
        id: list of activity id
    """
    assert os.path.exists(label_fname)

    # load time labels
    start, stop, id = [], [], []
    content = open(label_fname).read().splitlines()
    for line in content:
        # only retrieve action labels
        cnt = line.count('><')
        if cnt != 1:
            continue

        # parse action label
        toks = line[line.find('(')+1:line.find(')')].split('-')
        start.append(int(toks[0]))
        stop.append(int(toks[1]))
        id.append(line[1:line.find('>')])

    # double check overlapping
    for i in range(1, len(id)):
        if start[i] <= stop[i-1]:
            overlap = stop[i-1] - start[i]
            stop[i-1] += overlap // 2
            start[i] = stop[i-1] + 1
        assert start[i] > stop[i-1] and stop[i] > start[i], \
            '{}-->{}, followed by {}-->{}'.format(start[i-1], stop[i-1],
                                                  start[i], stop[i])
    return start, stop, id


def augment_label(start, stop, id, num_frames, bg_lbl):
    """Augment labels with background class

    Args:
        start: list of start frame id
        stop: list of stop frame id
        id: list of activity id
        num_frames: number of frames of that video
        bg_lbl: background label

    Returns:
        start_aug: augmented list of start frame id
        stop_aug: augmented list of stop frame id
        id_aug: augmented list of activity id
    """
    start_aug = []
    stop_aug = []
    id_aug = []

    # ignore background at the beginning
    # check the first index
    # if start[0] != 1:
        # start_aug.append(1)
        # stop_aug.append(start[0] - 1)
        # id_aug.append(bg_lbl)

    # run through the list
    N = len(id)
    for i in range(N):
        # add default start, stop, id
        start_aug.append(start[i])
        stop_aug.append(stop[i])
        id_aug.append(id[i])

        # check if there is a gap
        if i < N - 1:
            if stop[i] < start[i+1] - 1:
                start_aug.append(stop[i] + 1)
                stop_aug.append(start[i+1] - 1)
                id_aug.append(bg_lbl)

    # ignore background at the end
    # check the last index
    # if stop[-1] < num_frames:
        # start_aug.append(stop[-1] + 1)
        # stop_aug.append(num_frames)
        # id_aug.append(bg_lbl)
    return start_aug, stop_aug, id_aug


def main():
    args = parse_args()

    vid_list = os.listdir(args.framedir)
    vid_list.sort()

    _, inv_lbl_dict = load_label_dict(args.labeldict_pth)

    for vid in vid_list:
        # parse annotation
        label_fname = os.path.join(args.labeldir, vid+'.txt')
        start, stop, id = parse_label(label_fname, inv_lbl_dict)

        # augment annotation with background class
        frame_lst = glob.glob(os.path.join(args.framedir, vid, '*'+args.ext))
        frame_lst.sort()
        start, stop, id = augment_label(start, stop, id, len(frame_lst), args.bg_lbl)

        for i in range(len(id)):
            print(start[i], stop[i], id[i])

        # segment data
        print('segmenting video {}...'.format(vid))
        pbar = ProgressBar(max_value=len(id))
        for i in range(len(id)):
            folder = os.path.join(args.outputdir, vid,
                                  '{:02d}'.format(i+1) + '_' + id[i])

            if not os.path.exists(folder):
                os.makedirs(folder)

            for frame_id in range(start[i], stop[i]+1):
                frame_fn = '{:010d}{}'.format(frame_id, args.ext)
                src = os.path.join(args.framedir, vid, frame_fn)
                dst = os.path.join(folder, frame_fn)
                if not os.path.exists(src):
                    continue
                shutil.copy(src, dst)
            pbar.update(i)
    pass


if __name__ == '__main__':
    main()
