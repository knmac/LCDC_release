"""tfrecord writer for GTEA shopping dataset
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_utils.salads_writer_flow import SaladsWriterFlow


class GteaWriterFlow(SaladsWriterFlow):
    def generate_data_lst_from_split(self, split_fname):
        """ generate data list from the content of split_fname

        Args:
            split_fname: file name of split description for 50 salads

        Returns:
            data_lst
        """
        content = open(split_fname, 'r').read().splitlines()
        data_lst = [x for x in content if x != '']
        return data_lst
