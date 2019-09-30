"""tfrecord writer for 50 salads dataset
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import glob
import numpy as np
from data_utils.data_writer import DataWriter


class SaladsWriter(DataWriter):
    def _process_key(self, activity):
        """Return the actual activity class name
        """
        key = activity[activity.index('_')+1:]
        return key

    def _clsname2lbl(self, classname, lbl_dict):
        """Convert from classname to label id

        Args:
            classname: the classname to convert
            lbl_dict: label dictionary

        Returns:
            an integer, telling label of the given classname
        """
        return lbl_dict[classname]

    def generate_data_lst_from_split(self, split_fname):
        """Generate data list from the content of split_fname

        This may be different from dataset to dataset

        Args:
            split_fname: file name of split description for 50 salads

        Returns:
            data_lst
        """
        content = open(split_fname, 'r').read().splitlines()
        data_lst = ['rgb-'+x for x in content]
        return data_lst

    def load_im_pths(self, data_dir, data_lst, lbl_dict_pth, ext, frameskip):
        """
        Args:
            data_dir: where data are stored
            data_lst: list of data, each line is an activity video name,
                      e.g. ['rgb-01-1', 'rgb-03-2']
            lbl_dict_pth: path to label dictionary, containing all labels
            ext: image extenstion
            frameskip: how many frames to skip. Default fps is 30fps so if
                       frameskip=30, it means 1 frame/sec

        Returns:
            fname_pths: a list where each line is a sample
            labels: a list where each line is a label corresponding to a data
                    sample
        """
        return self.load_snippet_pths(data_dir, data_lst, lbl_dict_pth,
                                      snippet_len=1, stride=1, ext=ext,
                                      frameskip=frameskip)

    def load_snippet_pths(self, data_dir, data_lst, lbl_dict_pth, bg_lbl,
                          snippet_len, stride, ext, frameskip):
        """
        Args:
            data_dir: where data are stored
            data_lst: list of data, each line is an activity video name,
                      e.g. ['rgb-01-1', 'rgb-03-2']
            lbl_dict_pth: path to label dictionary, containing all labels
            bg_lbl: lable for background class
            snippet_len: number of frame per snippet
            stride: temporal stride to skip the frame
            ext: image extension
            frameskip: how many frames to skip. Default fps is 30fps so if
                       frameskip=30, it means 1 frame/sec

        Returns:
            fname_pths: a list where each line is a sample
            labels: a list where each line is a label corresponding to a data
                    sample
        """
        assert os.path.exists(data_dir)
        assert os.path.exists(lbl_dict_pth)
        assert type(data_lst) is list
        assert type(stride) is int
        assert stride > 0
        assert type(snippet_len) is int
        assert snippet_len > 0

        # retrieve all class labels
        lbl_list = open(lbl_dict_pth).read().splitlines()
        lbl_dict = {bg_lbl: 0}
        for i in range(len(lbl_list)):
            lbl_dict[lbl_list[i]] = i + 1

        # go through data_lst and retrieve fname_pths and labels
        fname_pths = []
        labels = []
        ignore_cnt = 0
        len_cnt = 0.
        for vid_id in data_lst:
            vid_dir = os.path.join(data_dir, vid_id)
            assert os.path.exists(vid_dir)
            activities = os.listdir(vid_dir)
            activities.sort()

            for activity in activities:
                # generate label from activity sub-folder name
                key = self._process_key(activity)
                if key not in lbl_dict:
                    continue
                label = self._clsname2lbl(key, lbl_dict)

                # retrieve list of png files in the sub-folder
                fnames = glob.glob(os.path.join(vid_dir, activity, '*.' + ext))
                fnames.sort()

                # downsample by each activity to make sure each snippet only
                # has one type of action. If we downsample outside this loop
                # one snippet may be multiple types of activity
                fnames = fnames[::frameskip]

                # ignore too short snippet
                N = len(fnames)
                if N < snippet_len:
                    ignore_cnt += 1
                    len_cnt += N
                    continue

                # append lists of file names and labels
                for i in range(0, N-snippet_len+1, stride):
                    snippet_fnames = fnames[i:i+snippet_len]
                    if len(snippet_fnames) == 1:
                        fname_pths.append(snippet_fnames[0])
                        labels.append(label)
                    else:
                        fname_pths.append(snippet_fnames)
                        labels.append(label)

        print('Ignore {} short snippets'.format(ignore_cnt))
        if ignore_cnt != 0:
            mean_len = len_cnt / ignore_cnt
            print('Mean length of ignored snippets: {:05f}'.format(mean_len))
        return fname_pths, labels

    def load_snippet_pths_mix(self, data_dir, data_lst, lbl_dict_pth, bg_lbl,
                              snippet_len, stride, ext, frameskip, purity=0.7):
        """
        Args:
            data_dir: where data are stored
            data_lst: list of data, each line is an activity video name,
                      e.g. ['rgb-01-1', 'rgb-03-2']
            lbl_dict_pth: path to label dictionary, containing all labels
            bg_lbl: lable for background class
            ext: image extension
            frameskip: how many frames to skip. Default fps is 30fps so if
                       frameskip=30, it means 1 frame/sec

        Returns:
            fname_pths: a list where each line is a sample
            labels: a list where each line is a label corresponding to a data
                    sample
        """
        assert os.path.exists(data_dir)
        assert os.path.exists(lbl_dict_pth)
        assert type(data_lst) is list

        # retrieve all class labels
        lbl_list = open(lbl_dict_pth).read().splitlines()
        lbl_dict = {bg_lbl: 0}
        for i in range(len(lbl_list)):
            lbl_dict[lbl_list[i]] = i + 1

        # go through data_lst and retrieve fname_pths and labels
        fname_pths = []
        labels = []
        cnt = 0
        for vid_id in data_lst:
            vid_dir = os.path.join(data_dir, vid_id)
            assert os.path.exists(vid_dir), '{} not found'.format(vid_dir)
            activities = os.listdir(vid_dir)
            activities.sort()

            fnames_per_vid = []
            labels_per_vid = []
            for activity in activities:
                fnames = glob.glob(os.path.join(vid_dir, activity, '*.'+ext))
                fnames.sort()
                key = self._process_key(activity)
                if key not in lbl_dict:
                    continue
                label = self._clsname2lbl(key, lbl_dict)

                fnames_per_vid += fnames
                labels_per_vid += [label] * len(fnames)

            # downsample
            fnames_per_vid = fnames_per_vid[::frameskip]
            labels_per_vid = labels_per_vid[::frameskip]

            # make mix class snippets
            N = len(fnames_per_vid)
            for i in range(0, N-snippet_len+1, stride):
                # find the representative label
                lbl_segment = labels_per_vid[i:i+snippet_len]
                values, counts = np.unique(lbl_segment, return_counts=True)
                index = np.argmax(counts)

                # ignore snippet if the representative label is not dominant
                if counts[index] < purity*counts.sum():
                    cnt += 1
                    continue

                # append data if the representative label is dominant enough
                fname_pths += [fnames_per_vid[i:i+snippet_len]]
                labels += [int(values[index])]
                pass
            pass
        print('Ignore {} impure snippets...'.format(cnt))
        return fname_pths, labels

    def load_snippet_pths_mid(self, data_dir, data_lst, lbl_dict_pth, bg_lbl,
                              snippet_len, stride, ext, frameskip):
        """
        Args:
            data_dir: where data are stored
            data_lst: list of data, each line is an activity video name,
                      e.g. ['rgb-01-1', 'rgb-03-2']
            lbl_dict_pth: path to label dictionary, containing all labels
            bg_lbl: lable for background class
            ext: image extension
            frameskip: how many frames to skip. Default fps is 30fps so if
                       frameskip=30, it means 1 frame/sec

        Returns:
            fname_pths: a list where each line is a sample
            labels: a list where each line is a label corresponding to a data
                    sample
        """
        assert os.path.exists(data_dir)
        assert os.path.exists(lbl_dict_pth)
        assert type(data_lst) is list

        # retrieve all class labels
        lbl_list = open(lbl_dict_pth).read().splitlines()
        lbl_dict = {bg_lbl: 0}
        for i in range(len(lbl_list)):
            lbl_dict[lbl_list[i]] = i + 1

        # go through data_lst and retrieve fname_pths and labels
        fname_pths = []
        labels = []
        for vid_id in data_lst:
            vid_dir = os.path.join(data_dir, vid_id)
            assert os.path.exists(vid_dir), '{} not found'.format(vid_dir)
            activities = os.listdir(vid_dir)
            activities.sort()

            fnames_per_vid = []
            labels_per_vid = []
            for activity in activities:
                fnames = glob.glob(os.path.join(vid_dir, activity, '*.'+ext))
                fnames.sort()
                key = self._process_key(activity)
                if key not in lbl_dict:
                    continue
                label = self._clsname2lbl(key, lbl_dict)

                fnames_per_vid += fnames
                labels_per_vid += [label] * len(fnames)

            # downsample
            fnames_per_vid = fnames_per_vid[::frameskip]
            labels_per_vid = labels_per_vid[::frameskip]

            # make mix class snippets
            N = len(fnames_per_vid)
            for i in range(0, N-snippet_len+1, stride):
                # find the representative label as the middle one
                lbl_segment = labels_per_vid[i:i+snippet_len]

                # append data and use the middle frame as groundtruth label
                fname_pths += [fnames_per_vid[i:i+snippet_len]]
                labels += [lbl_segment[snippet_len // 2]]
                pass
            pass
        return fname_pths, labels

    def load_snippet_pths_test(self, data_dir, data_lst, lbl_dict_pth, bg_lbl,
                               ext, frameskip):
        """
        Args:
            data_dir: where data are stored
            data_lst: list of data, each line is an activity video name,
                      e.g. ['rgb-01-1', 'rgb-03-2']
            lbl_dict_pth: path to label dictionary, containing all labels
            bg_lbl: lable for background class
            ext: image extension
            frameskip: how many frames to skip. Default fps is 30fps so if
                       frameskip=30, it means 1 frame/sec

        Returns:
            fname_pths: a list where each line is a sample
            labels: a list where each line is a label corresponding to a data
                    sample
        """
        assert os.path.exists(data_dir)
        assert os.path.exists(lbl_dict_pth)
        assert type(data_lst) is list

        # retrieve all class labels
        lbl_list = open(lbl_dict_pth).read().splitlines()
        lbl_dict = {bg_lbl: 0}
        for i in range(len(lbl_list)):
            lbl_dict[lbl_list[i]] = i + 1

        # go through data_lst and retrieve fname_pths and labels
        fname_pths = []
        labels = []
        for vid_id in data_lst:
            vid_dir = os.path.join(data_dir, vid_id)
            assert os.path.exists(vid_dir)
            activities = os.listdir(vid_dir)
            activities.sort()

            fnames_per_vid = []
            labels_per_vid = []
            for activity in activities:
                fnames = glob.glob(os.path.join(vid_dir, activity, '*.'+ext))
                fnames.sort()
                key = self._process_key(activity)
                if key not in lbl_dict:
                    continue
                label = self._clsname2lbl(key, lbl_dict)

                fnames_per_vid += fnames
                labels_per_vid += [label] * len(fnames)

            # downsample
            fnames_per_vid = fnames_per_vid[::frameskip]
            labels_per_vid = labels_per_vid[::frameskip]

            fname_pths += fnames_per_vid
            labels += labels_per_vid
        fname_pths = [[x] for x in fname_pths]
        return fname_pths, labels
