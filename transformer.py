import pandas as pd 
import numpy as np

class MinMaxNormalization:
    """
    Conduct normalization over pandas dataframe or numpy array
    """
    def __init__(self, X: pd.DataFrame or np.ndarray):
        self.X_min = None 
        self.X_max = None 
        self.X = X

    def fit(self):
        self.X_min = self.X.min()
        self.X_max = self.X.max()

    def transform(self, X):
        return (X - self.X_min) / (self.X_max - self.X_min)

    def back_transform(self, X_transformed):
        return X_transformed * (self.X_max - self.X_min) + self.X_min


