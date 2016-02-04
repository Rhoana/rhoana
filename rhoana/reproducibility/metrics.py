import numpy as np
from scipy.ndimage.filters import maximum_filter
import fast64counter

def thin_boundaries(im, mask):
    im = im.copy()
    assert (np.all(im >= 0)), "Label images must be non-negative"
    # repeatedly expand regions by one pixel until the background is gone
    while (im[mask] == 0).sum() > 0:
        zeros = (im == 0)
        im[zeros] = maximum_filter(im, 3)[zeros]
    return im

def Rand(pair, gt, pred, alpha):
    '''Parameterized Rand score

    Arguments are pairwise fractions, ground truth fractions, and prediction
    fractions.

    Equation 3 from Arganda-Carreras et al., 2015
    alpha = 0 is Rand-Split, alpha = 1 is Rand-Merge

    '''

    return np.sum(pair ** 2) / (alpha * np.sum(gt ** 2) +
                                (1.0 - alpha) * np.sum(pred ** 2))

def VI(pair, gt, pred, alpha):
    ''' Parameterized VI score

    Arguments are pairwise fractions, ground truth fractions, and prediction
    fractions.

    Equation 6 from Arganda-Carreras et al., 2015
    alpha = 0 is VI-Split, alpha = 1 is VI-Merge
    '''

    pair_entropy = - np.sum(pair * np.log(pair))
    gt_entropy = - np.sum(gt * np.log(gt))
    pred_entropy = - np.sum(pred * np.log(pred))
    mutual_information = gt_entropy + pred_entropy - pair_entropy
    return mutual_information / ((1.0 - alpha) * gt_entropy +
                                 alpha * pred_entropy)

def segmentation_metrics(ground_truth, prediction, seq=False):
    '''Computes adjusted FRand and VI between ground_truth and prediction.

    Metrics from: Crowdsourcing the creation of image segmentation algorithms
    for connectomics, Arganda-Carreras, et al., 2015, Frontiers in Neuroanatomy

    ground_truth - correct labels
    prediction - predicted labels

    Boundaries (label == 0) in prediction are thinned until gone, then are
    masked to foreground (label > 0) in ground_truth.

    Return value is ((FRand, FRand_split, FRand_merge), (VI, VI_split, VI_merge)).

    If seq is True, then it is assumed that the ground_truth and prediction are
    sequences that should be processed elementwise.

    '''

    # make non-sequences into sequences to simplify the code below
    if not seq:
        ground_truth = [ground_truth]
        prediction = [prediction]

    counter_pairwise = fast64counter.ValueCountInt64()
    counter_gt = fast64counter.ValueCountInt64()
    counter_pred = fast64counter.ValueCountInt64()

    for gt, pred in zip(ground_truth, prediction):
        mask = (gt > 0)
        pred = thin_boundaries(pred, mask)
        gt = gt[mask].astype(np.int32)
        pred = pred[mask].astype(np.int32)
        counter_pairwise.add_values_pair32(gt, pred)
        counter_gt.add_values_32(gt)
        counter_pred.add_values_32(pred)

    # fetch counts
    frac_pairwise = counter_pairwise.get_counts()[1]
    frac_gt = counter_gt.get_counts()[1]
    frac_pred = counter_pred.get_counts()[1]
    # normalize to probabilities
    frac_pairwise = frac_pairwise.astype(np.double) / frac_pairwise.sum()
    frac_gt = frac_gt.astype(np.double) / frac_gt.sum()
    frac_pred = frac_pred.astype(np.double) / frac_pred.sum()

    alphas = {'F-score': 0.5, 'split': 0.0, 'merge': 1.0}

    Rand_scores = {k: Rand(frac_pairwise, frac_gt, frac_pred, v) for k, v in alphas.items()}
    VI_scores = {k: VI(frac_pairwise, frac_gt, frac_pred, v) for k, v in alphas.items()}

    return {'Rand': Rand_scores, 'VI': VI_scores}
