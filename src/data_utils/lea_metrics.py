"""Metrics adopted from Lea et al
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import scipy.io


# -----------------------------------------------------------------------------
# Metrics functions
# -----------------------------------------------------------------------------
def accuracy(P, Y):
    """Framewise accuracy score

    P and Y are either lists of ndarray, or ndarray.
    - If list of ndarray: each entry correspond to a video
    - If ndarray: (predicted or groundtruth) labels of a video

    Args:
        P: prediction data
        Y: groundtruth data
    """
    if type(P) == list:
        return np.mean([np.mean(P[i] == Y[i]) for i in range(len(P))])*100
    else:
        return np.mean(P == Y) * 100
    return


def edit_score(P, Y, norm, bg_class):
    """Segmental edit score

    P and Y are either lists of ndarray, or ndarray.
    - If list of ndarray: each entry correspond to a video
    - If ndarray: (predicted or groundtruth) labels of a video

    Args:
        P: prediction data
        Y: groundtruth data
        norm: whether to normalize the scores
        bg_class: index of background class
    """
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = _segment_labels(P)
        Y_ = _segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c != bg_class]
            Y_ = [c for c in Y_ if c != bg_class]
    return _levenstein(P_, Y_, norm)


def overlap_f1(P, Y, n_classes, bg_class):
    """Ovelap F1 score with 10% overlapping

    P and Y are either lists of ndarray, or ndarray.
    - If list of ndarray: each entry correspond to a video
    - If ndarray: (predicted or groundtruth) labels of a video

    Args:
        P: prediction data
        Y: groundtruth data
        n_classes: number of classes
        bg_class: index of background class
    """
    overlap = 0.1
    if type(P) == list:
        tmp = [_overlap(P[i], Y[i], n_classes, bg_class, overlap)
               for i in range(len(P))]
        return np.mean(tmp)
    else:
        return _overlap(P, Y, n_classes, bg_class, overlap)


def mid_mAP(P, Y, S, bg_class):
    """mAP score with midpoint criterion

    P, Y, and S are lists of ndarray, each entry correspond to a video

    Args:
        P: prediction data
        Y: groundtruth data
        S: score data
        bg_class: index of background class

    Returns:
        Average precision of each class
        Mean of average precesion
    """
    # segment groundtruth intervals
    gt_file, gt_labels, gt_intervals = [], [], []
    for i, y in enumerate(Y):
        gt_labels += [_segment_labels(y)]
        gt_intervals += [np.array(_segment_intervals(y))]
        gt_file += [[i]*len(gt_labels[-1])]
    gt_file = np.hstack(gt_file)
    gt_intervals = np.vstack(gt_intervals)
    gt_labels = np.hstack(gt_labels)

    # segment detected intervals
    det_file, det_labels, det_intervals, det_scores = [], [], [], []
    for i, y in enumerate(P):
        det_labels += [_segment_labels(y)]
        det_intervals += [np.array(_segment_intervals(y))]
        det_file += [[i]*len(det_labels[-1])]
        det_scores += [S[i][inter[0]:inter[1]][:, label].max()
                       for inter, label in zip(det_intervals[-1],
                                               det_labels[-1])]
    det_file = np.hstack(det_file)
    det_intervals = np.vstack(det_intervals)
    det_labels = np.hstack(det_labels)
    det_scores = np.hstack(det_scores)

    pr, ap, mAP_mid = _midpoint_mAP(gt_file, det_file, gt_labels, det_labels,
                                    gt_intervals, det_intervals, det_scores,
                                    bg_class=bg_class)
    return np.array(ap), mAP_mid*100


# -----------------------------------------------------------------------------
# Local helper functions
# -----------------------------------------------------------------------------
def _segment_labels(Yi):
    """Segmenting the labels for segmental edit score
    """
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split


def _segment_intervals(Yi):
    """Segment the interval for F1 score
    """
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals


def _levenstein(p, y, norm=False):
    """Levenstein distance between 2 lists
    """
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j]+1, D[i, j-1]+1, D[i-1, j-1]+1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
    return score


def _overlap(p, y, n_classes, bg_class, overlap):
    """Overlap score with general overlap percentage
    """
    true_intervals = np.array(_segment_intervals(y))
    true_labels = _segment_labels(y)
    pred_intervals = np.array(_segment_intervals(p))
    pred_labels = _segment_labels(p)

    # Remove background labels
    if bg_class is not None:
        true_intervals = true_intervals[true_labels != bg_class]
        true_labels = true_labels[true_labels != bg_class]
        pred_intervals = pred_intervals[pred_labels != bg_class]
        pred_labels = pred_labels[pred_labels != bg_class]

    n_true = true_labels.shape[0]
    n_pred = pred_labels.shape[0]

    # We keep track of the per-class TPs, and FPs.
    # In the end we just sum over them though.
    TP = np.zeros(n_classes, np.float)
    FP = np.zeros(n_classes, np.float)
    true_used = np.zeros(n_true, np.float)

    for j in range(n_pred):
        # Compute IoU against all others
        intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) \
            - np.maximum(pred_intervals[j, 0], true_intervals[:, 0])
        union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) \
            - np.minimum(pred_intervals[j, 0], true_intervals[:, 0])
        IoU = (intersection / union)*(pred_labels[j] == true_labels)

        # Get the best scoring segment
        idx = IoU.argmax()

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        if IoU[idx] >= overlap and not true_used[idx]:
            TP[pred_labels[j]] += 1
            true_used[idx] = 1
        else:
            FP[pred_labels[j]] += 1

    TP = TP.sum()
    FP = FP.sum()
    # False negatives are any unused true segment (i.e. "miss")
    FN = n_true - true_used.sum()

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1 = 2 * (precision*recall) / (precision+recall)

    # If the prec+recall=0, it is a NaN. Set these to 0.
    F1 = np.nan_to_num(F1)
    return F1*100


def _midpoint_mAP(gt_files, det_files, gt_labels, det_labels, gt_intervals,
                  det_intervals, det_conf, bg_class=None):

    return _IoU_mAP(gt_files, det_files, gt_labels, det_labels, gt_intervals,
                    det_intervals, det_conf, bg_class=bg_class,
                    use_midpoint=True)


def _IoU_mAP(gt_files, det_files, gt_labels, det_labels, gt_intervals,
             det_intervals, det_conf, threshold=.5, bg_class=None,
             use_midpoint=False):
    """
    Returns:
        pr_all: precision-recall curves
        ap_all: AP for each class
        map: MAP
    """
    # Get labels. Remove background class if necessary
    labels = np.unique(gt_labels).tolist()
    if bg_class is not None and bg_class in labels:
        labels.remove(bg_class)
    n_labels = len(labels)

    if use_midpoint:
        overlap_fcn = _midpoint_criterion
        threshold = 0.5
    else:
        overlap_fcn = _interval_overlap

    pr_labels = []
    pr_all, rec_all, ap_all = [], [], []
    for i in range(n_labels):
        label = labels[i]
        if any(gt_labels == label):
            rec, prec, ap = _TH14eventdetpr(
                gt_files, det_files, gt_labels, det_labels, gt_intervals,
                det_intervals, label, det_conf, threshold, overlap_fcn)
            pr_labels += [label]
            pr_all += [prec]
            rec_all += [rec]
            ap_all += [ap]

    mAP = np.mean(ap_all)

    return pr_all, ap_all, mAP


def _midpoint_criterion(gt_inter, det_inter):
    """
    Args:
        gt_inter : truth
        det_inter : detection

    Return:
        '1' if detection midpoint is within truth, otherwise 0
    """
    ov = np.zeros([gt_inter.shape[0], det_inter.shape[0]], np.float)
    for i in range(gt_inter.shape[0]):
        midpoints = det_inter.mean(1)
        ov[i] = (midpoints >= gt_inter[i][0]) * (midpoints < gt_inter[i][1])
    return ov


def _interval_overlap(gt_inter, det_inter):
    ov = np.zeros([gt_inter.shape[0], det_inter.shape[0]], np.float)
    for i in range(gt_inter.shape[0]):
        union = np.maximum(gt_inter[i, 1], det_inter[:, 1]) - \
            np.minimum(gt_inter[i, 0], det_inter[:, 0])
        intersection = np.minimum(gt_inter[i, 1], det_inter[:, 1]) - \
            np.maximum(gt_inter[i, 0], det_inter[:, 0])
        intersection = intersection.clip(0, np.inf)
        ov[i] = intersection/union
    return ov


# Recreated from the THUMOS2014 code for computing mAP@k
def _TH14eventdetpr(gt_files, det_files,
                    gt_labels, det_labels,
                    gt_intervals, det_intervals, label, det_conf,
                    overlap, overlap_fcn=_interval_overlap):
    # returns recall, precision, ap

    videonames = np.unique(np.hstack([gt_files, det_files]))
    n_videos = len(videonames)

    unsortConf = []
    unsortFlag = []
    npos = float((gt_labels == label).sum())

    # Ensure there are predicted labels in detection list
    if not any(det_labels == label):
        return 0., 0., 0.

    for i in range(n_videos):
        gt = np.nonzero((gt_files == videonames[i])*(gt_labels == label))[0]
        det = np.nonzero((det_files == videonames[i])*(det_labels == label))[0]

        # If there are detections of this class:
        if det.shape[0] > 0:
            # Sort based on confidences. Most conf first
            ind_s = _argsort(-det_conf[det])
            det = det[ind_s]
            conf = det_conf[det]
            ind_free = np.ones(len(det))

            # If there are true segments of this class:
            if gt.shape[0] > 0:
                ov = overlap_fcn(gt_intervals[gt], det_intervals[det])
                for k in range(ov.shape[0]):
                    ind = np.nonzero(ind_free)[0]
                    if len(ind) > 0:
                        ind_m = np.argmax(ov[k][ind])
                        val_m = ov[k][ind][ind_m]
                        if val_m > overlap:
                            ind_free[ind[ind_m]] = 0

            ind1 = np.nonzero(ind_free == 0)[0]
            ind2 = np.nonzero(ind_free == 1)[0]

            # Mark '1' if true positive, '2' for false positive
            flag = np.hstack([np.ones(len(ind1)), 2*np.ones(len(ind2))])
            ttIdx = _argsort(np.hstack([ind1, ind2]))
            idx_all = np.hstack([ind1, ind2])[ttIdx]
            flagall = flag[ttIdx]

            unsortConf = np.hstack([unsortConf, conf[idx_all]])
            unsortFlag = np.hstack([unsortFlag, flagall])

    conf = np.vstack([np.hstack(unsortConf), np.hstack(unsortFlag)])

    idx_s = _argsort(-conf[0])
    tp = (conf[1][idx_s] == 1).cumsum()*1.
    fp = (conf[1][idx_s] == 2).cumsum()*1.
    is_correct = conf[1][idx_s] == 1
    rec = tp/float(npos)
    prec = tp/(fp+tp)
    ap = _prap(rec, prec, is_correct, npos)

    return rec, prec, ap


def _argsort(seq):
    """The numpy sort is inconistent with the matlab version for values that
    are the same
    """
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def _prap(rec, prec, tmp, npos):
    return prec[tmp == 1].sum() / npos


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def unit_test():
    """Unit test to make sure the code is working correctly"""
    # Fixed filename
    ROOT = '/home/knmac/projects/vid_time_model/data/EDTCN_results/50Salads/'\
           'mid/mid_motionconstraint_nomotion_g0/nepoch_200'
    RUN = 'run_11'
    SPLIT = 'Split_1'
    FNAME = os.path.join(ROOT, RUN, SPLIT+'.mat')

    # Load computed results
    content = open(os.path.join(ROOT, RUN, 'trials.txt')).read().splitlines()
    for line in content:
        if SPLIT in line:
            break
    tokens = line.split(' ')
    acc_rec = tokens[2].replace('accuracy:', '').replace(',', '')
    edit_rec = tokens[3].replace('edit_score:', '').replace(',', '')
    f1_rec = tokens[4].replace('overlap_f1:', '').replace(',', '')

    # Load data
    data = scipy.io.loadmat(FNAME)
    P, S, Y = data['P'].squeeze(), data['S'].squeeze(), data['Y'].squeeze()
    P = [x.squeeze() for x in P]
    S = S.tolist()
    Y = [x.squeeze() for x in Y]

    # Compute metrics
    acc = accuracy(P, Y)
    edit = edit_score(P, Y, norm=True, bg_class=0)
    f1 = overlap_f1(P, Y, n_classes=18, bg_class=0)
    _, mAP = mid_mAP(P, Y, S, bg_class=0)

    # Print out
    print('Testing metrics...')
    print('  Acc:   computed={:.02f} - recorded={}'.format(acc, acc_rec))
    print('  Edit:  computed={:.02f} - recorded={}'.format(edit, edit_rec))
    print('  F1@10: computed={:.02f} - recorded={}'.format(f1, f1_rec))
    print('  mAP:   computed={:.02f}'.format(mAP))
    return 0


if __name__ == '__main__':
    sys.exit(unit_test())
