import numpy as np
import pandas as pandas 

def merge_array(a,b):
    """
    Helper function for generate_k_fold_index()
    """
    c = list(a) + list(b)
    return np.array(c)

def generate_k_fold_index(X, y, k, shuffle=True, stratified=False, random_state=1):
    """
    Create k-fold index. 
    """
    assert len(X) == len(y)
    np.random.seed(random_state)
    cv_idx = X.index.values.copy()
    cv_dict = {'k':[], 'idx':[]}
    if shuffle == True:
        np.random.shuffle(cv_idx)
    if stratified == False:
        dist = len(cv_idx) // k
        for i in range(k-1):
            cv_dict['k'].append(i)
            cv_dict['idx'].append(cv_idx[i*dist: (i+1)*dist])
        cv_dict['k'].append(k-1)
        cv_dict['idx'].append(cv_idx[(k-1)*dist:])
    else:
        y = pd.Series(y)
        groups = y.unique()
        for group in groups:
            y_sub = y[y==group]
            cv_idx = y_sub.index.values.copy()
            dist = len(cv_idx) // k
            
            if len(cv_dict['k'])==0:
                for i in range(k-1):
                    cv_dict['k'].append(i)
                    cv_dict['idx'].append(cv_idx[i*dist: (i+1)*dist])
                cv_dict['k'].append(k-1)
                cv_dict['idx'].append(cv_idx[(k-1)*dist:])
            else:
                for i in range(k-1):
                    cv_dict['idx'][i] = merge_array(cv_idx[i*dist: (i+1)*dist], cv_dict['idx'][i])
                cv_dict['idx'][i] = merge_array(cv_idx[(k-1)*dist:], cv_dict['idx'][i])
                    

    return cv_dict