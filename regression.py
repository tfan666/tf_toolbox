import pandas as pd
import numpy as np
from scipy.linalg import solve
from scipy.stats import t
from joblib import delayed, Parallel
import scipy.optimize as opt

def least_square_beta_estimator(X, y, use_solver=False):
    if use_solver == False:
        beta = np.linalg.inv(X.T@X) @ X.T @y
    else:
        beta = solve(a=X.T @ X, b=X.T @ y)
    return beta

def least_square_loss(beta, X, y):
    n = X.shape[1]
    beta_l1 = np.matrix(beta).reshape(n,1)
    cost = (1/(2*n)) *np.power((y- X @ beta ),2).sum()
    return cost

def de_least_square(beta, X, y):
    n = X.shape[0]
    x0 = - 1/n * X.T @ (y - X @ beta)
    return x0

def gradient_descent(grad, loss, x0, X, y, lr=0.01, max_iter=10000, tol=0.01, return_hist=False):
    loss_hist = []
    loss_hist.append(loss(x0,X,y))
    for i in range(max_iter):
        x0 = x0 - lr * grad(x0,X,y)
        loss_hist.append(loss(x0,X,y))
        if abs(grad(x0,X,y)).max() < tol:
            break
    # print(f"Iter{i}| x: {x0.T}|de_loss: {grad(x0,X,y).T}")
    if return_hist == False:
        return x0
    else:
        return x0, loss_hist

def ridge_least_square_beta_estimator(X, y, _lambda):
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

def add_constant(X):
    n = X.shape[0]
    const = np.ones((n,1))
    X = np.concatenate([const, X], axis=1)
    return X

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
        self.fit_intercept = True
        self.gd_loss_hist = None
    
    def fit(self, method='OLS', fit_intercept=True, _lambda=1, lr=0.01):
        """
        This function fit the given data. Available methods:
            - OLS: simple linear regression (Orindary Least Square) using matrix multplication
            - OLS_solver: simple linear regression (Orindary Least Square) using linear solver
            - GD: Gradient Desecent
            - L1: Lasso Regression
            - L2: Ridge Regression
        """
        # check whether intercept is needed
        if fit_intercept == True:
            X = add_constant(self.X)
        else:
            X = self.X
            self.fit_intercept = False
        # check fitting methods
        if method == 'OLS':
            beta = least_square_beta_estimator(X=X, y=self.y)
            self.para['coef'] = beta
        elif method == 'OLS_solver':
            beta = least_square_beta_estimator(X=X, y=self.y, use_solver=True)
            self.para['coef'] = beta
        elif method == 'GD':
            x0 = np.ones((X.shape[1],1))
            beta, self.gd_loss_hist = gradient_descent(
                x0=x0, X=X, y=self.y, 
                grad=de_least_square, 
                loss=least_square_loss, 
                lr=lr, return_hist=True)
            self.para['coef'] = beta
        elif method == 'L1':
            x0 = np.zeros(X.shape[1])
            beta = l1_cost_minimize(x0=x0, _lambda=_lambda, X=X, y=y)
            self.para['coef'] = beta
        elif method == 'L2':
            y = self.y
            beta = ridge_least_square_beta_estimator(X=X, y=y, _lambda=_lambda)
            self.para['coef'] = beta
        else:
            print('Method not found')
        # update model class
        self.method = method
        self.residuals = np.array(self.y.reshape(self.n,)-self.predict(self.X)).flatten()
        self.RSS = (np.power(self.residuals,2)).sum()
        self.TSS = (np.power(self.y - self.y.mean(),2)).sum()
        self.r_squared = 1 - self.RSS/self.TSS
        self.sd_resid = np.power(self.RSS/(self.n-2),0.5)
        self.transform_dict = None

    def predict(self, new_X, interval=None, level=0.95, n_jobs=1):
        beta = self.para['coef']
        if self.fit_intercept == True:
            new_X = add_constant(new_X)
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
            if self.fit_intercept == True:
                X = add_constant(self.X)
            else:
                X = self.X
            x0 = new_X[i]
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
