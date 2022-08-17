import pandas as pd
import numpy as np
from scipy.linalg import solve
from scipy.stats import t, f
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

def sample_from_matrix(X, y, n_rows):
    idx = np.int64((np.random.random(size=n_rows)*len(X)).round())
    X_sample, y_sample = X[idx,], y[idx,]
    return X_sample, y_sample

def gradient_descent(
    grad, loss, x0, X, y, lr=0.01, max_iter=10000, tol=0.01, 
    return_hist=False, stochastic=False, sample_size=0.1):
    loss_hist = []
    if stochastic == False:
        loss_hist.append(loss(x0,X,y))
        for i in range(max_iter):
            x0 = x0 - lr * grad(x0,X,y)
            loss_hist.append(loss(x0,X,y))
            if abs(grad(x0,X,y)).max() < tol:
                break
    else:
        X_sample, y_sample = sample_from_matrix(X, y, int(sample_size*len(X)))
        loss_hist.append(loss(x0,X_sample,y_sample))
        for i in range(max_iter):
            x0 = x0 - lr * grad(x0,X_sample,y_sample)
            loss_hist.append(loss(x0,X_sample,y_sample))
            if abs(grad(x0,X_sample,y_sample)).max() < tol:
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
        self.sigma = None
        self.n = len(y)
        self.n_col = X.shape[1]
        self.beta_var = None 
        self.coef_se = None
        self.t_stats = None
        self.p_values = None
        self.k = None
        self.k_var = None
        self.f_stat =  None
        self.f_stat_p_value = None
        self.fit_intercept = True
        self.gd_loss_hist = None
    
    def fit(
        self, method='OLS', fit_intercept=True, 
        _lambda=1, lr=0.01, stochastic=False, sample_size=0.1):
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
            self.n_col += 1
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
                lr=lr, return_hist=True,
                stochastic=stochastic,
                sample_size=sample_size
                )
            self.para['coef'] = beta
        elif method == 'L1':
            y = self.y
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

        if method == 'OLS' or method == 'OLS_solver':
            self.sigma = np.sqrt((np.power(self.residuals, 2)).sum()/ (self.n - self.n_col))
            self.beta_var = np.linalg.inv(X.T @ X)*self.sigma**2 
            self.coef_se = np.diag(self.beta_var)**0.5
            self.t_stats = np.array(beta).flatten() / self.coef_se
            self.p_values = 2 * t.cdf(-abs(self.t_stats),self.n-self.n_col)
            if fit_intercept == True:
                self.k = np.concatenate(
                    [np.zeros((self.n_col-1,1)) , np.diag(np.ones(self.n_col-1))], 
                    axis=1)
            else:
                self.k = np.diag(np.ones(self.n_col))
            self.k_var = self.k @ np.linalg.inv(X.T @ X) @ self.k.T
            self.f_stat =  float((self.k @ beta).T @ np.linalg.inv(self.k_var) @ (self.k @ beta) / ((self.n_col-1) * self.sigma**2))
            self.f_stat_p_value = 1 - f.cdf(self.f_stat, self.n_col-1, self.n-self.n_col)

    def predict(self, new_X, interval=None, level=0.95, n_jobs=1):
        beta = self.para['coef']
        if self.fit_intercept == True:
            new_X = add_constant(new_X)
        y = new_X @ beta 
        y = np.array(y).reshape(self.n,)
        if interval == None:
            return y
        
        alpha = 1-level
        critical_value = t.ppf(1-alpha/2, self.n-2)
        if interval == 'confidence':
            SE = np.sqrt(self.sd_resid**2 * np.diag(new_X @ np.linalg.inv(new_X.T @ new_X) @ new_X.T))
            _type = 'ci'
        if interval == 'prediction':
            SE = np.sqrt(1 + self.sd_resid**2 * np.diag(new_X @ np.linalg.inv(new_X.T @ new_X) @ new_X.T))
            _type = 'pi'

        y_out = pd.DataFrame({
            'mean': y,
            f'lo_{level}_{_type}': y - critical_value * SE,
            f'hi_{level}_{_type}': y + critical_value * SE,
        })

        return y_out
    
    def summary(self, X_name=None):
        if self.method == 'OLS' or self.method == 'OLS_solver':
            beta = np.array(self.para['coef']).flatten()
            critical_value = t.ppf(1-0.05/2, self.n-2)
            model_summary = pd.DataFrame({
                'Estimate': np.array(beta).flatten(),
                'Std. Error': self.coef_se,
                't statistics': self.t_stats,
                'Pr(>|t|)': self.p_values,
                '[0.025': np.array(beta).flatten() - critical_value * self.coef_se,
                '0.975]': np.array(beta).flatten() + critical_value * self.coef_se
            })
            if X_name != None:
                model_summary.index = X_name
            return model_summary
        else:
            print('Only OLS has summary table for now.')