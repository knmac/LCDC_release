"""Build tfrecord files for different datasets
"""
import os
import sys
import argparse

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_utils import dataset_factory


def parseargs():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', help='data directory')
    parser.add_argument('--outputdir', help='directory storing outputs')
    parser.add_argument('--labelsdesc', help='labels description')
    parser.add_argument('--recordsize', type=int, default=100,
                        help='number of samples per tfrecord')
    parser.add_argument('--snippet_len', type=int,
                        help='number of frames per snippet')
    parser.add_argument('--stride', type=int,
                        help='stride for new snippet')
    parser.add_argument('--ext', help='image extension, e.g. png or jpg')
    parser.add_argument('--frameskip', type=int,
                        help='how many frames to skip for downsampling. This '
                             'is done before striding')
    parser.add_argument('--datasetname', type=str, default='50salads',
                        help='name of the dataset: `50salads`, `merl`')
    parser.add_argument('--mode', type=str, default='train',
                        help='`train` (shuffling data) or `test` (no shuffling)')
    parser.add_argument('--output_pattern', type=str,
                        help='prefix for output tfrecord, e.g. `train_mid_`')
    parser.add_argument('--splitlist', type=str,
                        help='file containing split description')
    parser.add_argument('--bg_lbl', type=str, default='background',
                        help='label for background action')
    parser.add_argument('--mix', type=int, default=0,
                        help='mix-class training, or not')
    parser.add_argument('--mid', type=int, default=0,
                        help='mix-class training, or not')
    parser.add_argument('--purity', type=float,
                        help='purity constraint for mix-class training')
    parser.add_argument('--dummy', type=int, default=0,
                        help='1 or 0, build dummy data to test with the first '
                             'samples')

    args = parser.parse_args()
    if args.mode == 'train':
        assert args.frameskip >= 0, 'frameskip must be positive'
    assert args.mode == 'train' or args.mode == 'test'
    assert args.ext != '', 'Require image extension'
    assert os.path.exists(args.splitlist)
    assert (args.dummy == 1) or (args.dummy == 0)
    return args


def main():
    """Main function"""
    # parser input arguments
    args = parseargs()

    # create file writer
    dataset_writer = dataset_factory.get_writer(args.datasetname)
    writer = dataset_writer()

    # retrieve list of data
    datalst = writer.generate_data_lst_from_split(args.splitlist)

    # decide whether to shuffle data
    if args.mode == 'train':
        to_shuffle = True
    elif args.mode == 'test':
        to_shuffle = False

    # reduce folders if processing for dummy data
    if args.dummy:
        datalst = [datalst[0]]

    # load data paths and labels
    if args.mode == 'train':
        if args.mid:
            fname_pths, labels = writer.load_snippet_pths_mid(
                args.datadir, datalst, args.labelsdesc, args.bg_lbl,
                snippet_len=args.snippet_len, stride=args.stride,
                ext=args.ext, frameskip=args.frameskip)
        elif args.mix:
            fname_pths, labels = writer.load_snippet_pths_mix(
                args.datadir, datalst, args.labelsdesc, args.bg_lbl,
                snippet_len=args.snippet_len, stride=args.stride,
                ext=args.ext, frameskip=args.frameskip, purity=args.purity)
        else:
            fname_pths, labels = writer.load_snippet_pths(
                args.datadir, datalst, args.labelsdesc, args.bg_lbl,
                snippet_len=args.snippet_len, stride=args.stride,
                ext=args.ext, frameskip=args.frameskip)
    elif args.mode == 'test':
        fname_pths, labels = writer.load_snippet_pths_test(
            args.datadir, datalst, args.labelsdesc, args.bg_lbl,
            ext=args.ext, frameskip=args.frameskip)
    print('Found {} samples...'.format(len(labels)))

    # reduce samples if processing for dummy data
    if args.dummy:
        fname_pths = fname_pths[:100]
        labels = labels[:100]

    # create tfrecords
    writer.write_record(fname_pths, labels, args.outputdir,
                        output_pattern=args.output_pattern,
                        to_shuffle=to_shuffle,
                        record_size=args.recordsize)

    # record the number of samples
    fname = os.path.join(args.outputdir, args.output_pattern+'_n_samples.txt')
    with open(fname, 'w') as f:
        f.write('{}\n'.format(len(labels)))
    pass


if __name__ == '__main__':
    main()
