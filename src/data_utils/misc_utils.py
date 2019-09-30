"""Miscellaneous utilities
"""
from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def visualize_flow(flow):
    """Visualize optical flow

    Args:
        flow: optical flow map with shape of (H, W, 2), with (y, x) order

    Returns:
        RGB image of shape (H, W, 3)
    """
    assert flow.ndim == 3
    assert flow.shape[2] == 2

    hsv = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 1], flow[..., 0])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_flow_sequence(flow_sequence):
    """Visualize a sequence of flow

    Args:
        flow_sequence: sequence of flow, ndarray of shape (N, H, W, 2)
    """
    assert flow_sequence.ndim == 4
    return np.array([visualize_flow(x) for x in flow_sequence])


def visualize_flow_sequence_batch(flow_batch):
    """Visualize a sequence of flow

    Args:
        flow_sequence: sequence of flow, ndarray of shape (batch_size, N, H, W, 2)
    """
    assert flow_batch.ndim == 5
    return np.array([visualize_flow_sequence(x) for x in flow_batch])


def compute_reduced_optical_flow(prev, next, out_h, out_w, flow=None,
                                 pyr_scale=0.5, levels=3, winsize=15,
                                 iterations=3, poly_n=5, poly_sigma=1.2,
                                 flags=0):
    """Compute optical flow from two images with reduction

    Args:
        prev: RGB frame at time (t-1), shape of (H, W, 3)
        next: RGB frame at time t, shape of (H, W, 3)
        out_h: output height
        out_w: output width

    Return:
        flow: optical flow of shape (out_h, out_w, 2) with (y, x) order.
            The flow is defined as x[t] - x[t-1], which is the reversed
            direction of the one from opencv.
    """
    # Convert to gray scale if needed
    prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)

    assert prev.shape == next.shape
    H, W = prev.shape

    # Compute dense optical flow at the original resolution using opencv
    flow = cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels,
                                        winsize, iterations, poly_n,
                                        poly_sigma, flags)

    # Convert from (x, y) to (y, x) order
    # Reverse the flow direction to match the definition of our rDeRF
    flow = -flow[:, :, ::-1]

    # Compute downscale ratio
    assert H % out_h == 0, 'H not divided by out_h'
    assert W % out_w == 0, 'W not divided by out_w'
    assert (H // out_h) == (W // out_w), 'h and w downscale ratio are different'
    ratio = H // out_h

    # Reduce in spatial dimension
    flow = flow[::ratio, ::ratio, :]
    assert (flow.shape[0] == out_h) and (flow.shape[1] == out_w)

    # Reduce in magnitude to compensate
    flow /= ratio
    return flow


def compute_optical_flow_sequence(frame_lst, out_h, out_w):
    """Compute optical flow from a list of frame

    Args:
        frame_lst: ndarray of frames after mean removal, shape of (N, H, W, 3)
        out_h: output height
        out_w: output width

    Return:
        flow_lst: optical flow ndarray, shape of (N-1, out_h, out_w, 2)
    """
    # Recover after mean removal
    frame_lst[..., 0] += _R_MEAN
    frame_lst[..., 1] += _G_MEAN
    frame_lst[..., 2] += _B_MEAN
    frame_lst = frame_lst.astype(np.uint8)

    # compute reduced optical flow for consecutive frames
    N = len(frame_lst)
    flow_lst = np.zeros([N-1, out_h, out_w, 2], dtype=np.float32)
    for t in range(1, N):
        flow = compute_reduced_optical_flow(frame_lst[t-1], frame_lst[t],
                                            out_h, out_w)
        flow_lst[t-1] = flow
    return flow_lst


def compute_optical_flow_sequence_batch(snippet_batch, out_h, out_w):
    """Compute optical flow from a batch of video snippet. Each snippet is a
    list of frame in sequence.

    Args:
        snippet_batch: batch of video snippet
        out_h: output height
        out_w: output width

    Return:
        flow_batch: batch of corresponding flow
    """
    flow_batch = [compute_optical_flow_sequence(x, out_h, out_w) for x in snippet_batch]
    return np.array(flow_batch)
