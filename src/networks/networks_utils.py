""" This file contains general utitilies for networks
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import tensorflow as tf
import matplotlib.cm
import numpy as np
import pickle


def load_pretrained(data_path, ignore_missing=False, extension='npy',
                    encoding='latin1', showinfo=True, initoffset=True):
    """ Load pretrained weights as *.npy files

    Args:
        data_path: path to the pretrained model
        ignore_missing: ignore missing weights from the network
        extension: either npy (numpy) or pkl (pickle)
        encoding: data encoding of the weights
        showinfo: whether to show which keys are assigned or ignored
        initoffset: whether to initialize offsets with nonezero values

    Returns:
        assign_ops: assignment ops between network layers and loaded weights.
                    Need sess.run(assign_ops) to actually assign.
    """
    assert os.path.exists(data_path)
    assert extension == 'npy' or extension == 'pkl'

    # retrieve data
    if extension == 'npy':
        data_dict = np.load(data_path, encoding=encoding).item()
    else:
        data_dict = pickle.load(open(data_path, 'rb'))

    # append assign_ops
    print('\n'+'='*80)
    print('Loading pretrained parameters from ' + data_path + '...')
    assign_ops = []
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                # try to get the variable
                try:
                    var = tf.get_variable(subkey)
                    if showinfo:
                        print('get variable in the network: ' + key + '/' + subkey)
                except ValueError:
                    if showinfo:
                        # key exists in loaded data but not in our network
                        # therefore ignore
                        print('--> ignore1 ' + key + '/' + subkey)
                    if not ignore_missing:
                        raise
                    else:
                        continue

                # try to assign the values
                try:
                    assign_ops.append(var.assign(data_dict[key][subkey]))
                    if showinfo:
                        print('assign pretrain model ' + subkey + ' to ' + key)
                except ValueError:
                    # fix the case of having different channels order for
                    # conv and dconv operations
                    net_shape = tuple(var.shape.as_list())
                    trained_shape = data_dict[key][subkey].shape

                    if subkey == 'weights' and len(trained_shape) == 4:
                        trained_perm_shape = (trained_shape[3], trained_shape[2],
                                              trained_shape[0], trained_shape[1])
                        if net_shape == trained_perm_shape:
                            trained_param = data_dict[key][subkey]
                            trained_param_perm = np.transpose(
                                    trained_param, [3, 2, 0, 1])
                            if showinfo:
                                print('--> permutate and assign ' + subkey +
                                      ' to ' + key)
                            assign_ops.append(var.assign(trained_param_perm))
                        else:
                            # key exists in loaded data AND our network, BUT
                            # weights cannot be assigned because of different
                            # shape --> random init
                            xinit = _random_init(var)
                            assign_ops.append(tf.assign(var, xinit))
                            # if showinfo:
                                # print('--> ignore2 ' + key + '/' + subkey)
                            # if not ignore_missing:
                                # raise

                    if (subkey == 'biases') and (net_shape != trained_shape):
                        xinit = _random_init(var)
                        assign_ops.append(tf.assign(var, xinit))

    # initialize offsets
    print('Initializing offsets...')
    for x in tf.global_variables():
        # ignore non-offset keys
        if 'offset' not in x.op.name:
            # print('--> not initializing ' + x.op.name + ' in loading pretrained')
            continue

        # ignored keys that were assigned with pretrained data or keys that
        # do not follow the convention `key/subkey`, e.g. variables generated
        # by momentum optimizer
        try:
            key, subkey = x.op.name.split('/')
        except ValueError:
            continue
        if key in data_dict.keys() and subkey in data_dict[key].keys():
            continue

        # initialize offsets to zero or uniformly on [-0.05..0.05]
        xshape = x.shape.as_list()
        if initoffset:
            print('--> randomly initializing ' + x.op.name)
            xinit = np.random.random(xshape) * 0.1 - 0.05
        else:
            print('--> initializing ' + x.op.name + ' as zeros')
            xinit = np.zeros(xshape)
        assign_ops.append(tf.assign(x, xinit))
    return assign_ops


def _random_init(var):
    xshape = var.shape.as_list()
    print('--> randomly initializing ' + var.op.name)
    xinit = np.random.random(xshape) * 0.1 - 0.05
    return xinit


def colorize_tensor(value, vmin=None, vmax=None, cmap=None, extend=False):
    """ A utility function for TensorFlow that maps a grayscale image to a
    matplotlib colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before
    mapping to a grayscale colormap.

    Args:
        value: 2D Tensor of shape [height, width] or 3D Tensor of shape
               [height, width, 1].
        vmin: the minimum value of the range used for normalization.
              (Default: value minimum)
        vmax: the maximum value of the range used for normalization.
              (Default: value maximum)
        cmap: a valid cmap named for use with matplotlib's `get_cmap`.
              (Default: 'gray')
        extend: whether to extend the tensor from 3-D to 4-D

    Returns:
        3-D or 4-D tensor of shape [height, with, 3] or [1, height, with, 3]

    Reference:
        https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
    """
    # normalize the whole matrix to vmin..vmax
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # normalize each row so that each row sum to 1
    row_sum = tf.expand_dims(tf.reduce_sum(value, axis=1), axis=1)
    value = value / row_sum

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    if extend:
        value = tf.expand_dims(value, axis=0)
    return value


def viz_flow(flow):
    """Visualize optical flow

    Args:
        flow: optical flow map with shape of (H, W, 2), with (y, x) order

    Returns:
        RGB image of shape (1, H, W, 3)
    """
    mag, ang = _cart2polar(flow[..., 1], flow[..., 0])

    # hsv = tf.zeros([flow.shape[0], flow.shape[1], 3], dtype=tf.uint8)
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 1] = 255
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    h = ang*180/np.pi/2
    s = tf.zeros([flow.shape[0], flow.shape[1]], dtype=tf.float32) + 255
    v = mag - tf.reduce_min(mag)
    v = v * 255 / tf.reduce_max(v)
    hsv = tf.stack([h, s, v], axis=2)
    rgb = tf.image.hsv_to_rgb(hsv)
    rgb = tf.expand_dims(rgb, axis=0)
    return rgb


def _cart2polar(y, x):
    """Convert from cartesian to polar coordinates
    """
    mag = tf.sqrt(y**2 + x**2)
    ang = tf.atan(y / x)
    return mag, ang
