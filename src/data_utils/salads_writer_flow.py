"""tfrecords writer for 50Salads, with flow data
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
import numpy as np
from data_utils.salads_writer import SaladsWriter


class SaladsWriterFlow(SaladsWriter):
    def _read_img(self, fname):
        """Overload here to negate the use of _read_img in data_writer
        """
        raise NotImplemented

    def _read_flow(self, fname):
        """Read a single flow from a file name

        Args:
            fname: npy filename containing the flow

        Return:
            flow: read-in optical flow
        """
        assert os.path.exists(fname)
        flow = np.load(fname)
        assert flow.shape[2] == 2, 'Only allow 2-channel flows'
        return flow

    def _build_feature(self, fname_pths, lbl):
        """Overload _build_feature, from data_writer

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
            img = self._read_flow(fname_pths)
            feat = {'label': self._int64_feature(lbl),
                    'image': self._bytes_feature(tf.compat.as_bytes(
                        img.tostring()))}
        else:
            N = len(fname_pths)

            # retrive image size to allocate snippet
            h, w, k = self._read_flow(fname_pths[0]).shape
            snippet = np.zeros([N, h, w, k], dtype=np.float32)
            for i in range(N):
                snippet[i] = self._read_flow(fname_pths[i])
            feat = {'label': self._int64_feature(lbl),
                    'snippet': self._bytes_feature(tf.compat.as_bytes(
                        snippet.tostring()))}
        return feat
