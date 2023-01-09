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

def log_loss(y_prob, y_true):
    log_loss = -(y*np.log(p) + (1-y)*np.log(1-p)).mean()
    return log_loss

def gini_impurity(X:np.ndarray or list):
    """
    Compute Gini Impurity 
    X: can be either numpy array or python list
    """
    if isinstance(X, list):
        X = np.array(X)

    K = np.unique(X)
    cnt = len(X)
    gini_impurity = 1
    for k in K:
        pk = np.sum(([X==k]))/cnt
        gini_impurity -= pk ** 2 
    return gini_impurity

def entropy_impurity(X:np.ndarray or list):
    """
    Compute Entropy Impurity 
    X: can be either numpy array or python list
    """
    if isinstance(X, list):
        X = np.array(X)

    K = np.unique(X)
    cnt = len(X)
    entropy_impurity = 0
    for k in K:
        pk = np.sum(([X==k]))/cnt
        entropy_impurity -= pk * np.log2(pk)
    return entropy_impurity

def area(a):
    """
    Helper function for IoU()
    """
    assert a[0] < a[2]
    assert a[1] > a[3]
    return abs((a[2] - a[0]) * (a[3] - a[1]))
area(a)

def interact_area(a,b):
    """
    Helper function for IoU()
    """
    # make sure this is a rectangular
    assert a[0] < a[2]
    assert a[1] > a[3]
    assert b[0] < b[2]
    assert b[1] > b[3]
    # check if there is any overlap areas
    if a[2] <= b[0] or a[0] >= b[2] or a[3] >= b[1] or a[1] <= b[3]:
        return 0
    else:
        x = [
            max(a[0], b[0]),
            min(a[1], b[1]),
            min(a[2], a[2]),
            max(a[3], a[3])
        ]
        return area(x)

def IoU(a,b):
    """
    Intersect Over Union:
        - a,b : list of four elements [top left x, top left y, bottom right x, bottom right y] 
    """
    assert a[0] < a[2]
    assert a[1] > a[3]
    assert b[0] < b[2]
    assert b[1] > b[3]
    return interact_area(a,b)/(area(a) + area(b) - interact_area(a,b)) 
