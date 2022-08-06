from sklearn.metrics import classification_report, brier_score_loss, log_loss, recall_score, precision_score, accuracy_score
import sys
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd


def f2_score(y_true=None, y_pred=None, precision=None, recall=None, type='direct'):
    """
    Harmonic weighted mean of precision and recall, with
    more weight on recall.
    """
    if type=='direct':
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f2_score = (1+ 2**2) / ((2**2/recall) + 1/precision)
        return f2_score

    if type=='indirect':
        f2_score = (1+ 2**2) / ((2**2/recall) + 1/precision)
        return f2_score
    else:
        sys.exit("type has to be one of 'direct' or 'indirect'.")