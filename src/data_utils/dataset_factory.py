"""Create data reader and writer for different datasets
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_utils.salads_reader import SaladsReader
from data_utils.salads_reader_flow import SaladsReaderFlow
from data_utils.gtea_reader import GteaReader
from data_utils.gtea_reader_flow import GteaReaderFlow

from data_utils.salads_writer import SaladsWriter
from data_utils.salads_writer_flow import SaladsWriterFlow
from data_utils.gtea_writer import GteaWriter
from data_utils.gtea_writer_flow import GteaWriterFlow


def get_reader(datasetname):
    """ Get reader given datasetname

    Args:
        datasetname: name of the dataset

    Returns:
        dataset reader
    """
    reader_dict = {
        '50salads': SaladsReader,
        '50salads_flow': SaladsReaderFlow,
        'gtea': GteaReader,
        'gtea_flow': GteaReaderFlow,
    }
    assert datasetname in reader_dict, \
        'Dataset not supported: {}'.format(datasetname)
    return reader_dict[datasetname]


def get_writer(datasetname):
    """ Get reader given datasetname

    Args:
        datasetname: name of the dataset

    Returns:
        dataset reader
    """
    writer_dict = {
        '50salads': SaladsWriter,
        '50salads_flow': SaladsWriterFlow,
        'gtea': GteaWriter,
        'gtea_flow': GteaWriterFlow,
    }
    assert datasetname in writer_dict, \
        'Dataset not supported: {}'.format(datasetname)
    return writer_dict[datasetname]
