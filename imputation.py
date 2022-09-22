import pandas as pd
import numpy as np

def MICE(df, estimator, first_fill='mean', iterations=10, return_type='data'):
    """
    df: pandas dataframe
    estimator: preditecive model used to impute data
    first_fill: the inital imputation methods, must of one of "mean" or "median"
    iterations: num of iterations 
    return_type: "data" or "data+loss"

    """
    if first_fill == 'mean':
        df0 = df.fillna(df.mean()).copy()
        df_ = df.fillna(df.mean()).copy()
    elif first_fill == 'median':
        df0 = df.fillna(df.median()).copy()
        df_ = df.fillna(df.median()).copy()
    for __ in range(iterations):
        for col in df.columns:
            col_missing_idx = df[df[col].isnull()].index.values
            subset = df_.iloc[df.index.difference(col_missing_idx)]
            X_train, y_train = subset.drop(columns=col), subset[col]
            X_test = df_.drop(columns=col).iloc[col_missing_idx]
            imputer = estimator()
            imputer.fit(X=np.array(X_train), y=np.array(y_train))
            y_pred = imputer.predict(X_test)
            df_[col].loc[col_missing_idx] = y_pred
        loss = df0 - df_
    if return_type == 'data':
        return df_
    elif return_type == 'data+loss':
        return df_, loss
