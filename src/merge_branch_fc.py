"""Merge two branches using fc layer
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import scipy.io as sio
import glob
import argparse

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_classes', type=int, default=11,
        help='Number of classes (including background class)')
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Init learning rate')
    parser.add_argument(
        '--decay', type=float, default=3e-5,
        help='Decaying rate')
    parser.add_argument(
        '--batch', type=int, default=512,
        help='Batch size')
    parser.add_argument(
        '--n_epochs', type=int, default=1,
        help='Number of epochs')
    parser.add_argument(
        '--input_shape', type=int, default=1024+4096,
        help='Input dimension')
    parser.add_argument(
        '--output_shape', type=int, default=1024,
        help='Output dimension')

    parser.add_argument(
        '--dataset', type=str,
        default='gtea', choices=['50salads', 'gtea'],
        help='Name of dataset')
    parser.add_argument(
        '--input_dir', type=str,
        default='./data/GTEA/tcnfeat/2stream_vaniapp/Split_1',
        help='Input directory')
    parser.add_argument(
        '--output_dir', type=str,
        default='./data/GTEA/tcnfeat/2stream_vaniapp_fc1024/Split_1',
        help='Output directory')
    parser.add_argument(
        '--savedmodel_dir', type=str,
        default='./data/GTEA/tcnfeat',
        help='Where to save the merging model')
    parser.add_argument(
        '--train_split', type=str,
        default='./data/GTEA/splits/Split_1/train.txt',
        help='Training split')
    parser.add_argument(
        '--test_split', type=str,
        default='./data/GTEA/splits/Split_1/test.txt',
        help='Testing split')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # if not os.path.isdir(args.savedmodel_dir):
    #     os.makedirs(args.savedmodel_dir)

    assert os.path.isfile(args.train_split)
    assert os.path.isfile(args.test_split)
    return args


def load_data(input_dir, split, dataset):
    """Load data
    """
    # Retriev file list
    split_content = open(split).read().splitlines()
    fname_lst = os.listdir(input_dir)
    if dataset == '50salads':
        fname_lst = [os.path.join(input_dir, x) for x in fname_lst
                     if x.replace('.avi.mat', '').replace('rgb-', '')
                     in split_content]
    elif dataset == 'gtea':
        fname_lst = [os.path.join(input_dir, x) for x in fname_lst
                     if x.replace('.avi.mat', '') in split_content]
    fname_lst.sort()

    # Read data
    x, y = [], []
    for fname in fname_lst:
        assert os.path.isfile(fname)
        data = sio.loadmat(fname)
        x.append(data['A'])
        y.append(data['Y'])
    x = np.vstack(x)
    y = np.vstack(y).squeeze()
    assert x.shape[0] == y.shape[0]
    return x, y


def extract_feature(extractor, input_dir, output_dir):
    """Extract features and save
    """
    fname_lst = glob.glob(os.path.join(input_dir, '*.avi.mat'))
    fname_lst.sort()

    for fname in fname_lst:
        data = sio.loadmat(fname)
        fusion = extractor([data['A']])[0]
        assert fusion.shape[0] == data['A'].shape[0]
        mdict = {
            'A': fusion,
            'Y': data['Y']
        }
        sio.savemat(os.path.join(output_dir, os.path.basename(fname)), mdict)
    pass


def main():
    """Main function"""
    # Load data
    x_train, y_train = load_data(args.input_dir, args.train_split, args.dataset)
    x_test, y_test = load_data(args.input_dir, args.test_split, args.dataset)

    y_train_1hot = np_utils.to_categorical(y_train, args.n_classes)
    y_test_1hot = np_utils.to_categorical(y_test, args.n_classes)

    # Build model
    model = Sequential()
    model.add(Dense(args.output_shape, input_shape=[args.input_shape],
                    activation='sigmoid', name='fc1'))
    model.add(Dense(args.n_classes, activation='softmax', name='score'))

    optim = optimizers.Adam(lr=args.lr, decay=args.decay)
    model.compile(optimizer=optim, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Train model
    model.fit(x_train, y_train_1hot, shuffle=True,
              batch_size=args.batch, epochs=args.n_epochs,
              verbose=1, validation_data=(x_test, y_test_1hot))
    test_loss, test_acc = model.evaluate(x_test, y_test_1hot)
    print('Test acc = {:.02f}'.format(test_acc * 100))
    # model.save(os.path.join(args.savedmodel_dir, 'fusion_vani_1024.h5'))

    # Use model to fuse
    extractor = K.function([model.layers[0].input], [model.layers[0].output])

    # Save features
    extract_feature(extractor, args.input_dir, args.output_dir)
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main())
