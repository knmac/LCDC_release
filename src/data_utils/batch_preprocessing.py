from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
import numpy as np


# mean image from ImageNet
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _concat_batch(data):
    """ Concatenate data from (N, H, W, C) to (H, W, C*N)

    Args:
        data: data to concatenate, shape of (N, H, W, C) or (H, W, C)

    Returns:
        data_con: concatenated data, shape (H, W, C*N) or (H, W, C) if shape of
            data is (H, W, C)
        C: number of channels in data
        N: number of samples in data, None if data has shape of (H, W, C)
    """
    assert data.shape.ndims == 3 or data.shape.ndims == 4, \
        'Only allows tensors of 3 or 4 dimensions'
    if data.shape.ndims == 3:
        H, W, C = data.shape.as_list()
        return data, C, None

    N, H, W, C = data.shape.as_list()
    data_tran = tf.transpose(data, [1, 2, 3, 0])
    data_con = tf.reshape(data_tran, [H, W, C * N])
    return data_con, C, N


def _unconcat_batch(data_con, C, N):
    """ Unconcatenate data from (H, W, C*N) to (N, H, W, C)

    Args:
        data_con: concatenated data, shape of (H, W, N*C)
        C: number of channels
        N: number of samples. If N is None, return data_r

    Returns:
        Unconcatenated data of shape (N, H, W, C) or (H, W, C) if N is None.
    """
    if N is None:
        return data_con

    H, W, CN = data_con.shape.as_list()
    assert CN == C * N
    data_tran = tf.reshape(data_con, [H, W, C, N])
    data = tf.transpose(data_tran, [3, 0, 1, 2])
    return data


def mean_removal(data):
    """ Remove mean image (ImageNet) from data

    Args:
        data: tensor of shape (w, h, 3) or (n, w, h, 3)

    Return:
        mean removed data as tf.float32
    """
    assert data.shape.ndims == 3 or data.shape.ndims == 4, \
        'Only allows tensors of 3 or 4 dimensions'
    mean_img = np.zeros(data.shape.as_list())

    mean_img[..., 0] = _R_MEAN
    mean_img[..., 1] = _G_MEAN
    mean_img[..., 2] = _B_MEAN
    return tf.cast(data, tf.float32) - mean_img


def random_flip_left_right(data):
    """ Randomly flip an image or batch of image left/right uniformly

    Args:
        data: tensor of shape (H, W, C) or (N, H, W, C)

    Returns:
        Randomly flipped data
    """
    data_con, C, N = _concat_batch(data)
    data_con = tf.image.random_flip_left_right(data_con)
    return _unconcat_batch(data_con, C, N)


def _crop(data, offset_height, offset_width, crop_height, crop_width):
    """ Crops the data using provided offsets and sizes

    Args:
        data: tensor of shape (H, W, C)
        offset_height: a scalar tensor for the height offset
        offset_width: a scalar tensor for the width offset
        crop_height: the height of cropped image
        crop_width: the width of cropped image

    Returns:
        the cropped (and resized) image
    """
    original_shape = data.shape
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    assert original_shape[0] >= crop_height
    assert original_shape[1] >= crop_width
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    data = tf.slice(data, offsets, cropped_shape)

    return tf.reshape(data, cropped_shape)


def random_crop(data, crop_height, crop_width):
    """Crop the given data randomly

    Args:
        data: tensor of shape (N, H, W, C) or (H, W, C)
        crop_height: the new height
        crop_width: the new width

    Return:
        Randomly cropped data
    """
    data_con, C, N = _concat_batch(data)

    # check cropping size
    data_H, data_W, _ = data_con.shape.as_list()
    assert data_H >= crop_height
    assert data_W >= crop_width

    # maximum offset to crop
    max_offset_height = tf.reshape(data_H - crop_height + 1, [])
    max_offset_width = tf.reshape(data_W - crop_width + 1, [])

    # randomize offset to crop
    offset_height = tf.random_uniform([], maxval=max_offset_height,
                                      dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width,
                                     dtype=tf.int32)

    # crop data
    data_crop = _crop(data_con, offset_height, offset_width, crop_height,
                      crop_width)
    return _unconcat_batch(data_crop, C, N)


def central_crop(data, crop_height, crop_width):
    """Crop the given data at the center

    Args:
        data: tensor of shape (N, H, W, C) or (H, W, C)
        crop_height: the new height
        crop_width: the new width

    Return:
        Cropped data
    """
    data_con, C, N = _concat_batch(data)
    data_H, data_W, _ = data_con.shape.as_list()
    assert data_H >= crop_height
    assert data_W >= crop_width

    # centralize offset to crop
    offset_height = (data_H - crop_height) / 2
    offset_width = (data_W - crop_width) / 2

    # crop data
    data_crop = _crop(data_con, offset_height, offset_width, crop_height,
                      crop_width)
    return _unconcat_batch(data_crop, C, N)


def aspect_preserving_resize_smallest_side(data, smallest_side):
    """ Resize the data based on the smallest side

    For example: (480, 640, 3) --> (224, 298, 3)

    Args:
        data: 4D or 3D tensor (N, H, W, C) or (H, W, C)
        smallest_side: size of the smallest side after resizing

    Returns:
        resized image
    """
    # get the shape of input
    assert data.shape.ndims == 3 or data.shape.ndims == 4, \
        'Only allows tensors of 3 or 4 dimensions'
    if data.shape.ndims == 3:
        H, W, _ = data.shape.as_list()
    else:
        _, H, W, _ = data.shape.as_list()

    # resize data so that the smallest size is the same as target size
    if W < H:
        new_w = smallest_side
        scale = new_w * 1.0 / W
        new_h = int(scale * H)
    else:
        new_h = smallest_side
        scale = new_h * 1.0 / H
        new_w = int(scale * W)
    data = tf.image.resize_images(data, [new_h, new_w])
    return data


def aspect_preserving_resize_zeropad(data, largest_side):
    """ Resize the data based on the largest side and zero pad the other

    For example: (480, 640, 3) --> (224, 224, 3)
    ------------------
    |                |
    |                |
    |                |
    |                |
    |                |
    ------------------
    to
    -----------
    |000000000|
    |         |
    |         |
    |000000000|
    -----------

    Args:
        data: 4D or 3D tensor (N, H, W, C) or (H, W, C)
        largest_side: size of the largest side after resizing. The other side
            will be zero padded

    Returns:
        resized data
    """
    # get the shape of input
    assert data.shape.ndims == 3 or data.shape.ndims == 4, \
        'Only allows tensors of 3 or 4 dimensions'
    if data.shape.ndims == 3:
        H, W, _ = data.shape.as_list()
    else:
        _, H, W, _ = data.shape.as_list()

    # resize data so that the largest size is the same as target size
    if H < W:
        new_w = largest_side
        scale = new_w * 1.0 / W
        new_h = int(scale * H)
    else:
        new_h = largest_side
        scale = new_h * 1.0 / H
        new_w = int(scale * W)
    data = tf.image.resize_images(data, [new_h, new_w])

    # pad the image with zeros. Since we already resized the inputs wrt
    # the largest side, this operation only pads zeros
    data = tf.image.resize_image_with_crop_or_pad(data, largest_side, largest_side)
    return data


def preprocess_for_train(data, output_height, output_width, resize_side_min,
                         resize_side_max, resize_mode='distort'):
    """ Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

    Args:
        data: 4D or 3D tensor (N, H, W, C) or (H, W, C).
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    if data.shape.ndims == 3:
        H, W, _ = data.shape.as_list()
    else:
        _, H, W, _ = data.shape.as_list()

    if output_height != H and output_width != W:
        resize_side = int(np.random.uniform(resize_side_min, resize_side_max))
        if resize_mode == 'distort':
            data = tf.image.resize_images(data, [resize_side, resize_side])
        elif resize_mode == 'preserve_crop':
            data = aspect_preserving_resize_smallest_side(data, resize_side)
        elif resize_mode == 'preserve_pad':
            data = aspect_preserving_resize_zeropad(data, resize_side)
    data = random_crop(data, output_height, output_width)
    # data = random_flip_left_right(data)
    return mean_removal(data)


def preprocess_for_test(data, output_height, output_width, resize_side,
                        resize_mode='distort'):
    """Preprocesses the given image for evaluation.

    Args:
        data: 4D or 3D tensor (N, H, W, C) or (H, W, C)
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side: The smallest side of the image for aspect-preserving
            resizing.

    Returns:
        A preprocessed image.
    """
    if data.shape.ndims == 3:
        H, W, _ = data.shape.as_list()
    else:
        _, H, W, _ = data.shape.as_list()

    if output_height != H and output_width != W:
        if resize_mode == 'distort':
            data = tf.image.resize_images(data, [resize_side, resize_side])
        elif resize_mode == 'preserve_crop':
            data = aspect_preserving_resize_smallest_side(data, resize_side)
        elif resize_mode == 'preserve_pad':
            data = aspect_preserving_resize_zeropad(data, resize_side)
    data = central_crop(data, output_height, output_width)
    return mean_removal(data)


def preprocess(data, output_height, output_width, is_training,
               resize_side_min=None, resize_side_max=None):
    """Preprocesses the given image batch.

    Args:
        data: 4D or 3D tensor (N, H, W, C) or (H, W, C)
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        is_training: `True` if we're preprocessing the image for training and
            `False` otherwise.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing. If `is_training` is `False`, then this
            value is used for rescaling. If None, then this value is set as
            the smallest side
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing. If `is_training` is `False`, this value
            is ignored. Otherwise, the resize side is sampled from
            [resize_size_min, resize_size_max]. If None, then this value is set
            as the smallest_side +10%

    Returns:
        A preprocessed image.
    """
    # get the shape of input
    assert data.shape.ndims == 3 or data.shape.ndims == 4, \
        'Only allows tensors of 3 or 4 dimensions'
    if data.shape.ndims == 3:
        H, W, _ = data.shape.as_list()
    else:
        _, H, W, _ = data.shape.as_list()

    # set resize_side if not set
    smallest_side = min(output_height, output_width)
    if resize_side_min is None:
        resize_side_min = smallest_side
    if resize_side_max is None:
        resize_side_max = smallest_side + smallest_side//10

    # preprocess according to training or not
    if is_training:
        return preprocess_for_train(data, output_height, output_width,
                                    resize_side_min, resize_side_max)
    else:
        return preprocess_for_test(data, output_height, output_width,
                                   resize_side_min)
    pass
