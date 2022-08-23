import numpy as np
import pandas as pandas 

def generate_k_fold_index(X, y, k, shuffle=True, random_state=1):
    assert len(X) == len(y)
    np.random.seed(random_state)
    cv_idx = X.index.values.copy()
    if shuffle == True:
        np.random.shuffle(cv_idx)
    dist = len(cv_idx) // k
    cv_dict = {'k':[], 'idx':[]}
    for i in range(k-1):
        cv_dict['k'].append(i)
        cv_dict['idx'].append(cv_idx[i*dist: (i+1)*dist])
    cv_dict['k'].append(k-1)
    cv_dict['idx'].append(cv_idx[(k-1)*dist:])
    return cv_dict