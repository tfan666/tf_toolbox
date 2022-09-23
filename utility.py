import pandas as pd
import numpy as np

def gini(X:np.ndarray or list):
    """
    Compute Gini Index 
    X: can be either numpy array or python list
    """
    if isinstance(X, list):
        X = np.array(X)

    K = np.unique(X)
    cnt = len(X)
    gini = 1
    for k in K:
        pk = np.sum(([X==k]))/cnt
        gini -= pk ** 2 
    return gini

def get_day_of_week(date, output_string = False):
    if output_string == False:
        return pd.Timestamp(date).isoweekday()
    else:
        return pd.Timestamp(date).day_name()
    
 
