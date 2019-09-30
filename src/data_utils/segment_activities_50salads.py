"""Segment frames of a video in 50 salad datasets to activity
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
from progressbar import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framedir', type=str,
                        help='Location of extracted frames')
    parser.add_argument('--annodir', type=str,
                        help='Annotation directory')
    parser.add_argument('--outputdir', type=str,
                        help='Output directory')
    parser.add_argument('--ext', type=str,
                        help='image extension')
    parser.add_argument('--level', type=str,
                        help='granuity level')

    args = parser.parse_args()

    assert os.path.exists(args.framedir)
    assert os.path.exists(args.annodir)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    return args


def main():
    args = parse_args()
    assert args.level in ['low', 'mid', 'eval', 'high']

    # retrieve list of all videos
    vid_list = os.listdir(args.framedir)
    vid_list.sort()

    # go through each video
    for vid in vid_list:
        # get annnotation
        anno_fn = os.path.join(args.annodir, vid.replace('rgb-', '')+'.txt')
        anno_content = open(anno_fn).read().splitlines()

        # process the first item
        frame_id, label = anno_content[0].split(' ')
        frame_id = int(frame_id)
        segment_id = 1
        folder = os.path.join(args.outputdir, vid,
                              '{:02d}_{}'.format(segment_id, label))
        if not os.path.exists(folder):
            os.makedirs(folder)
        frame_fn = 'frame_{:07d}{}'.format(frame_id, args.ext)
        src = os.path.join(args.framedir, vid, frame_fn)
        dst = os.path.join(folder, frame_fn)
        shutil.copy(src, dst)
        old_label = label

        # segment data
        print('segmenting video {}...'.format(vid))
        pbar = ProgressBar(max_value=len(anno_content))
        for i in range(1, len(anno_content)):
            line = anno_content[i]
            frame_id, label = line.split(' ')
            frame_id = int(frame_id)

            if old_label != label:
                old_label = label
                segment_id += 1
                folder = os.path.join(args.outputdir, vid,
                                      '{:02d}_{}'.format(segment_id, label))
                if not os.path.exists(folder):
                    os.makedirs(folder)

            frame_fn = 'frame_{:07d}{}'.format(frame_id, args.ext)
            src = os.path.join(args.framedir, vid, frame_fn)
            dst = os.path.join(folder, frame_fn)
            if not os.path.exists(src):
                continue
            shutil.copy(src, dst)
            pbar.update(i)
        pass
    pass


if __name__ == '__main__':
    main()
