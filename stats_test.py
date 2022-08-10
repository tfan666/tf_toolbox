import pandas as pd
import numpy as np
from scipy.stats import t


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

def poisson(x, _lambda):
        fx = _lambda**x * np.exp(1)**(-_lambda) / np.math.factorial(x)
        return fx 
