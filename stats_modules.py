import pandas as pd
import numpy as np




def least_square(X, y):
    beta = np.linalg.inv(X.T@X) @ X.T @y
    beta_0 = (X @ beta - y).mean()
    return beta, beta_0

class linear_regression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None
        self.beta_0 = None
        self.method = None
    
    def fit(self, method='OLS'):
        if method == 'OLS':
            self.beta, self.beta_0 = least_square(X=self.X, y=self.y)
        elif method == 'GD':
            print('Gridient Desecnt WIP')
        elif method == 'L1':
            print('L1 WIP')
        elif method == 'L2':
            print('L2 WIP')
        self.method = method

    def predict(self, new_X):
        if self.method == 'OLS':
            y = new_X @ self.beta + self.beta_0
            return y

        

def poisson(x, _lambda):
        fx = _lambda**x * np.exp(1)**(-_lambda) / np.math.factorial(x)
        return fx 