import tensorflow as tf
from random import shuffle
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from progressbar import ProgressBar
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class DataWriter(object):
    def __init__(self):
        pass

    # Abstract methods---------------------------------------------------------
    @abc.abstractmethod
    def load_im_pths(self):
        return

    @abc.abstractmethod
    def load_snippet_pths(self):
        return

    # Public methods-----------------------------------------------------------
    def write_record(self, fname_pths, labels, output_dir, output_pattern,
                     to_shuffle=False, record_size=5000):
        """ Write tfrecord for images

        Args:
            fname_pths: a list where each item is a string (path of an image)
                        or a list of string (frames of a snippet)
            labels: a list of labels, must be the same length as fname_pths
            output_dir: a string, where tfrecord files are generated
            output_pattern: a string, pattern of the tfrecord file name
            to_shuffle: boolean, whether fname_pths and labels should shuffle.
                        the relative positions between fname_pths and labels is
                        kept after shuffling
            record_size: number of images per tfrecord file
        """
        assert type(fname_pths) is list
        assert type(labels) is list
        assert len(fname_pths) == len(labels)

        # make output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # shuffle list if needed
        if to_shuffle:
            zipped = list(zip(fname_pths, labels))
            shuffle(zipped)
            fname_pths, labels = zip(*zipped)

        # making tfrecord files
        print('Making tfrecord...')
        writer = None
        N = len(labels)
        pbar = ProgressBar(max_value=N)
        record_cnt = 0
        for i in range(N):
            # split into multiple files to avoid big tfrecords
            if i % record_size == 0:
                # close opened writer
                # unopened writer means the first writer
                if writer is not None:
                    writer.close()

                # prepare new writer
                record_cnt += 1
                record_fname = os.path.join(
                        output_dir, output_pattern + \
                        '_{:05d}.tfrecord'.format(record_cnt))
                writer = tf.python_io.TFRecordWriter(record_fname)

            # build feature
            feat = self._build_feature(fname_pths[i], labels[i])

            # write feature to tfrecord file
            example = tf.train.Example(features=tf.train.Features(
                feature=feat))
            writer.write(example.SerializeToString())

            # update progress bar
            pbar.update(i)

        pbar.update(N)
        writer.close()
        pass

    # Private methods----------------------------------------------------------
    def _build_feature(self, fname_pths, lbl):
        """ Build feature from a single image or a snippet

        Args:
            fname_pths: a string (for single image mode) or a list of string
                        (for snippet mode)
            lbl: a single integer

        Returns:
            feat: dictionary of feature
        """
        assert (type(fname_pths) is str) or (type(fname_pths) is list), \
                'fname_pths must be either a string or a list'
        assert type(lbl) is int

        if type(fname_pths) is str:
            img = self._read_img(fname_pths)
            feat = {'label': self._int64_feature(lbl),
                    'image': self._bytes_feature(tf.compat.as_bytes(
                        img.tostring()))}
        else:
            N = len(fname_pths)

            # retrive image size to allocate snippet
            h, w, k = self._read_img(fname_pths[0]).shape
            snippet = np.zeros([N, h, w, k], dtype=np.uint8)
            for i in range(N):
                snippet[i] = self._read_img(fname_pths[i])
            feat = {'label': self._int64_feature(lbl),
                    'snippet': self._bytes_feature(tf.compat.as_bytes(
                        snippet.tostring()))}
        return feat

    def _read_img(self, im_pth):
        """ Read a single image

        Args:
            im_pth: full path of the image to read

        Returns:
            img: raw image content
        """
        assert os.path.exists(im_pth)
        img = io.imread(im_pth)
        assert img.shape[2] == 3, 'Only allow 3-channel images'
        return img

    def _int64_feature(self, value):
        """ Convert value to int64 feature for TfRecord

        Args:
            value: value to convert
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """ Convert value to bytes feature for TfRecord

        Args:
            value: value to convert
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
