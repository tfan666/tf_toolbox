import pandas as pd
import numpy as np




def least_square(X, y):
    beta = np.linalg.inv(X.T@X) @ X.T @y
    beta_0 = (X @ beta).mean() - y.mean()
    return beta, beta_0

class linear_regression:
    "This class stores methods for linear regressions"
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.para = {}
        self.method = None
    
    def fit(self, method='OLS'):
        """
        This function fit the given data. Available methods:
            - OLS: simple linear regression (Orindary Least Square)
            - GD:Gradient Desecent
            - L1: Lasso Regression
            - L2: Ridge Regression
        """
        if method == 'OLS':
            
            beta, beta_0 = least_square(X=self.X, y=self.y)
            self.para['coef'] = beta
            self.para['intercept'] = beta_0
        elif method == 'GD':
            print('Gridient Desecnt WIP')
        elif method == 'L1':
            print('L1 WIP')
        elif method == 'L2':
            print('L2 WIP')
        self.method = method

    def predict(self, new_X):
        if self.method == 'OLS':
            beta = self.para['coef']
            beta_0 = self.para['intercept']
            y = new_X @ beta + beta_0
            return y

def poisson(x, _lambda):
        fx = _lambda**x * np.exp(1)**(-_lambda) / np.math.factorial(x)
        return fx 