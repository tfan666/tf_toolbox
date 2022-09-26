from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd  

def sample_df(X, y, sample_rate, random_state=1):
    """
    Helper function to sample dataframe.
    """
    sample_cnt = int(sample_rate * len(X))
    rs_X = X.sample(sample_cnt, random_state=random_state)
    rs_y = y[rs_X.index]
    return rs_X, rs_y


def combined_feature_selector(
    X: np.ndarray, y=None, estimators=[],
    sample_rate=.8, random_state=1, prefit=True,
    max_features=X.shape[1], threshold=None
    ):

    """
    Function to select features when all feature selector agrees

    Parameters:
        - X: predictors 
        - y: labels
        - estimators: list object of all the constructured estimator objects
        - sample rate: sample rate of each subset drawn from main dataset that used to fit estimators
        - random_state: seed 
        - prefit: 
        - max_features: max features num can be selected, default to all features in
        - threshold: 

    Output:
        - list that contains the name of feature selected
    """

    # make sure there are more than 1 estimator
    assert len(estimators) > 1

    # train the first selector
    rs_X, rs_y = sample_df(X, y, sample_rate=sample_rate, random_state=random_state)
    estimator0 = estimators[0]
    kernal0 = estimator0.fit(rs_X, rs_y)
    selector0 = SelectFromModel(
        kernal0, 
        prefit=prefit, 
        max_features=max_features,
        threshold=threshold)
    feature_out0 = selector0.get_support()

    # train the rest of selectors and combine the features
    for estimator in estimators[1:]:
        random_state += 1
        rs_X, rs_y = sample_df(X, y, sample_rate=sample_rate, random_state=random_state)
        kernal = estimator.fit(rs_X, rs_y)
        selector = SelectFromModel(
            kernal, 
            prefit=prefit, 
            max_features=max_features,
            threshold=threshold)
        feature_out = selector.get_support()

        for i in range(len(feature_out0)):
            if feature_out0[i] == True and feature_out[i] == False:
                feature_out0[i] == False
    return X.columns[feature_out0]

def find_binary_split(X: np.ndarray, y: np.ndarray, impurity_func: callable):
    """
    Function that find the best binary split that maximizes the information gain.

    Params:
        - X: features
        - y: labels
        - impurity_func: information gain function to compute impurity

    Output:
        - binary bin threshold
    """
    df = pd.DataFrame({
        'X': X,
        'y': y
    })

    info_gain_base = 0
    for node in np.unique(df['X']).round():
        cut = node + .5
        info_gain = impurity_func(df['y']) - (impurity_func(df[df['X'] >cut]['y']) + impurity_func(df[df['X'] <cut]['y']))
        if info_gain > info_gain_base:
            info_gain_base = info_gain
            node_out = cut
    return node_out
