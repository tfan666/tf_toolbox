import pandas as pd
import numpy as np

def least_square(X, y):
    beta = np.linalg.inv(X.T@X) @ X.T @y
    beta_0 = (X @ beta - y).mean()
    return beta, beta_0

def poisson(x, _lambda):
        fx = _lambda**x * np.exp(1)**(-_lambda) / np.math.factorial(x)
        return fx 