"""tfrecord reader for GTEA dataset
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_utils.salads_reader import SaladsReader


class GteaReader(SaladsReader):
    pass
