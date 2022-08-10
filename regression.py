import pandas as pd
import numpy as np
from scipy.linalg import solve
from scipy.stats import t
from joblib import delayed, Parallel

def least_square(X, y, use_solver=False):
    if use_solver == False:
        beta = np.linalg.inv(X.T@X) @ X.T @y
    else:
        beta = solve(a=X.T @ X, b=X.T @ y)
    return beta

def ridge_least_square(X, y, _lambda):
    beta = np.linalg.inv(X.T@X + _lambda * np.identity(len(X.T@X))) @ X.T @y
    return beta

def l1_cost_function(beta_l1, _lambda, X, y):
    beta_l1 = np.matrix(beta_l1).reshape(X.shape[1],1)
    cost = (1/2)*np.power((y - X @ beta_l1 ),2).sum() + _lambda * abs(beta_l1).sum()
    return cost

def l1_cost_minimize(x0, _lambda, X, y):
    x0 = np.matrix(x0).reshape(len(x0),1)

    res = opt.minimize(
        fun=l1_cost_function, 
        options={'maxiter':100000},
        # jac=grad,
        args=(_lambda, X,y),
        x0=x0)
    beta_l1 = res['x']
    return beta_l1

class linear_regression:
    "This class stores methods for linear regressions"
    def __init__(self, X, y):
        if isinstance(X, np.matrix) == True:
            self.X = X
        else:
            self.X = np.matrix(X)
        
        self.y = y
        self.para = {}
        self.method = None
        self.residuals = None
        self.RSS = None
        self.r_squared = None
        self.TSS = None
        self.sd_resid = None
        self.n = len(y)
    
    def fit(self, method='OLS', _lambda=1):
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
            X = self.X
            y = self.y
            x0 = np.zeros(X.shape[1])
            beta = l1_cost_minimize(x0=x0, _lambda=_lambda, X=X, y=y)
            self.para['coef'] = beta
        elif method == 'L2':
            X = self.X
            y = self.y
            beta = ridge_least_square(X=X, y=y, _lambda=_lambda)
            self.para['coef'] = beta
        else:
            print('Method not found')
        # update model class
        self.method = method
        self.residuals = self.y-self.predict(self.X)
        self.RSS = (np.power(self.residuals,2)).sum()
        self.TSS = (np.power(self.y - self.y.mean(),2)).sum()
        self.r_squared = 1 - self.RSS/self.TSS
        self.sd_resid = np.power(self.RSS/(self.n-2),0.5)

    def predict(self, new_X, interval=None, level=0.95, n_jobs=1):
        beta = self.para['coef']
        y = new_X @ beta 
        y = np.array(y).reshape(self.n,)
        if interval == None:
            return y
        
        y_out = {
            'mean': [],
            'lo': [],
            'hi': []
        }
        alpha = 1-level
        critical_value = t.ppf(1-alpha/2, self.n-2)

        def report(i, interval):
            x0 = new_X[i]
            X = self.X
            if interval == 'confidence':
                SE_ = np.float64(np.sqrt(self.sd_resid**2 * x0 @ np.linalg.inv(X.T @ X) @ x0.T))
            if interval == 'prediction':
                SE_ = np.float64(np.sqrt(1 + self.sd_resid**2 * x0 @ np.linalg.inv(X.T @ X) @ x0.T))
            hi = y[i] + critical_value*SE_
            lo = y[i] - critical_value*SE_
            report = {
                'mean': y[i],
                'lo': lo,
                'hi': hi
            }
            return report

        if interval=='confidence':
            r = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(report)(
                i, interval = 'confidence') 
                for i in range(len(new_X)) 
                )

        if interval=='prediction':
            r = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(report)(
                i, interval = 'prediction') 
                for i in range(len(new_X)) 
                )

        return pd.DataFrame(r)

