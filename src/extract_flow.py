"""Extract optical flow for multiple datasets using OpenCV
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse
import numpy as np
import skimage.io
import skimage.transform
import cv2


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--frameskip', type=int, help='frameskip for downsampling')
    parser.add_argument('--activity_dir', type=str, help='path to activity directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--target_height', type=int, default=256)
    parser.add_argument('--target_width', type=int, default=256)

    args = parser.parse_args()
    return args


def compute_optflow(fname1, fname2):
    """Compute optical flow from two images

    Args:
        fname1: filename of the previous frame
        fname2: filename of the next frame

    Return:
        flow: optical flow, shape of (H, W, 2), the first channel is x
              dimension, the second channel is y dimension
    """
    # Read images
    im1 = skimage.io.imread(fname1)
    im2 = skimage.io.imread(fname2)

    # Compute optical flow
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    im1 = skimage.transform.resize(
        im1, [args.target_height, args.target_width], preserve_range=True,
        mode='constant', anti_aliasing=True)
    im2 = skimage.transform.resize(
        im2, [args.target_height, args.target_width], preserve_range=True,
        mode='constant', anti_aliasing=True)
    flow = cv2.calcOpticalFlowFarneback(
        im1, im2, flow=None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # convert from (x, y) to (y, x) order
    # import ipdb; ipdb.set_trace()
    # import matplotlib.pyplot as plt
    # flow_yx = flow[:, :, ::-1]
    # flow_rgb = visualize_flow(flow_yx)
    return flow


def main():
    """Main function"""
    # Get the list of videos
    vid_lst = os.listdir(args.activity_dir)
    vid_lst.sort()

    # For each video in the list
    for vid_id in vid_lst:
        print(vid_id)
        if not os.path.exists(os.path.join(args.output_dir, vid_id)):
            os.makedirs(os.path.join(args.output_dir, vid_id))

        # Get the list of all segments
        seg_lst = os.listdir(os.path.join(args.activity_dir, vid_id))
        seg_lst.sort()

        # For each segment in the list
        for seg_id in seg_lst:
            if not os.path.exists(os.path.join(args.output_dir, vid_id, seg_id)):
                os.makedirs(os.path.join(args.output_dir, vid_id, seg_id))

            # Get the list of frames
            frame_lst = os.listdir(os.path.join(args.activity_dir, vid_id, seg_id))
            frame_lst.sort()

            # Downsample
            frame_lst = frame_lst[::args.frameskip]
            N = len(frame_lst)

            # Compute optical flow on consecutive frames
            prefix_in = os.path.join(args.activity_dir, vid_id, seg_id)
            prefix_out = os.path.join(args.output_dir, vid_id, seg_id)
            if N == 1:
                flow = np.zeros([args.target_height, args.target_width, 2], dtype=np.float32)
                out_fname = os.path.join(prefix_out, frame_lst[0].replace('.png', '.npy'))
                np.save(out_fname, flow)
            else:
                for i in range(N-1):
                    fname1 = os.path.join(prefix_in, frame_lst[i])
                    fname2 = os.path.join(prefix_in, frame_lst[i+1])
                    flow = compute_optflow(fname1, fname2)
                    out_fname = os.path.join(prefix_out, frame_lst[i].replace('.png', '.npy'))
                    np.save(out_fname, flow)
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
