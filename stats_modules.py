import pandas as pd
import numpy as np
from scipy.linalg import solve

def least_square(X, y, use_solver=False):
    if use_solver == False:
        beta = np.linalg.inv(X.T@X) @ X.T @y
    else:
        beta = solve(a=X.T @ X, b=X.T @ y)
    return beta

def wald_test(p_hat, p_0, n):
    t = (p_hat - p_0) / (np.sqrt(p_hat *(1-p_hat)/n)) 
    return t

def binominal_test(p_hat, p_0, n):
    t = (p_hat - p_0) / (np.sqrt(p_0 *(1-p_0)/n)) 
    return t

def wald_test_interval(p_hat, Z, p_0, n):
    return p_hat + np.array([-1,1]) * Z* np.sqrt(p_hat * (1 - p_hat) /n)

def score_test_interval(p_hat, Z, p_0, n):
    t = p_hat * (n/(n+Z**2)) + 1/2 * (Z**2/(n+Z**2)) + np.array([-1,1]) *\
    Z * np.sqrt(1/(n+Z**2) * (p_hat *(1-p_hat) * (n/n+Z**2) + 1/4 * (Z**2/n+Z**2)))
    return t

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
            
            beta = least_square(X=self.X, y=self.y)
            self.para['coef'] = beta
        elif method == 'OLS_solver':
            beta = least_square(X=self.X, y=self.y, use_solver=True)
            self.para['coef'] = beta
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
            y = new_X @ beta
            return y

def poisson(x, _lambda):
        fx = _lambda**x * np.exp(1)**(-_lambda) / np.math.factorial(x)
        return fx 