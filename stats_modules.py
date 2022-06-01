import pandas as pd
import numpy as np
from scipy.linalg import solve




def least_square(X, y, use_solver=False):
    if use_solver == False:
        beta = np.linalg.inv(X.T@X) @ X.T @y
    else:
        beta = solve(a=X.T @ X, b=X.T @ y)
    beta_0 = (X @ beta).mean() - y.mean()
    return beta, beta_0

class linear_regression:
    "This class stores methods for linear regressions"
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.para = {}
        self.method = None
        self.residuals = None
    
    def fit(self, method='OLS'):
        """
        This function fit the given data. Available methods:
            - OLS: simple linear regression (Orindary Least Square) using matrix multplication
            - OLS_solver: simple linear regression (Orindary Least Square) using linear solver
            - GD:Gradient Desecent
            - L1: Lasso Regression
            - L2: Ridge Regression
        """
        # check fitting methods
        if method == 'OLS':
            
            beta, beta_0 = least_square(X=self.X, y=self.y)
            self.para['coef'] = beta
            self.para['intercept'] = beta_0
        elif method == 'OLS_solver':
            beta, beta_0 = least_square(X=self.X, y=self.y, use_solver=True)
            self.para['coef'] = beta
            self.para['intercept'] = beta_0
        elif method == 'GD':
            print('Gridient Desecnt WIP')
        elif method == 'L1':
            print('L1 WIP')
        elif method == 'L2':
            print('L2 WIP')
        else:
            print('Method not found')
        # update model class
        self.method = method
        self.residuals = self.y-self.predict(self.X)

    def predict(self, new_X):
        if self.method == 'OLS' or self.method == 'OLS_solver':
            beta = self.para['coef']
            beta_0 = self.para['intercept']
            y = new_X @ beta + beta_0
            return y

def poisson(x, _lambda):
        fx = _lambda**x * np.exp(1)**(-_lambda) / np.math.factorial(x)
        return fx 