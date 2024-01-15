import torch

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_EER(scores, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100