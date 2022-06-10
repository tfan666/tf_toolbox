from joblib import load, dump, Parallel, delayed
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import numpy as np
import pandas as pd

def get_n_spilt_index(data, n_spilit=5):
    index_list_in = np.arange((len(data)))
    output_index_out = []
    index_len = len(index_list_in)
    step = round(index_len/n_spilit)
    for i in range(n_spilit):
        output_index_out.append(i*step)
    return output_index_out

class ETS:
    def __init__(self, y):
        self.y = y
        self.best_para = None
        self.n_jobs = -2
        self.para_search_records = None
        self.model = None
        self.alpha = 0.05
        self.ts_cv = 5

    def get_tscv_results(self, error, trend, seasonal, seasonal_periods):
        y = self.y
        y = y.interpolate().dropna()
        n_spilit = self.ts_cv
        idx = get_n_spilt_index(y, n_spilit)

        output = []
        for i in range(n_spilit):
            train, test = y[: idx[i]], y[idx[i]:]
            model = ETSModel(
                endog=y, 
                error=error, 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods).fit(full_output=False, disp=False)

            pred = model.forecast(len(test))
            rmse = mean_squared_error(y_true=test, y_pred=pred)**(0.5)
            output.append(rmse)
        output = np.array(output)
        mean_rmse = output.mean()
        sd_rmse = output.std()
        output_dict = {
            'mean_rmse': mean_rmse,
            'std_rmse': sd_rmse,
            'error': error,
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods}
        return output_dict

    def search_para(self, para, n_jobs = -2):
        self.n_jobs = n_jobs
        r = Parallel(n_jobs=n_jobs)(
            delayed(self.get_tscv_results)(
                error, trend, seasonal, seasonal_periods ) 
                for error in para['error'] \
                    for trend in para['trend'] \
                        for seasonal in para['seasonal'] \
                            for seasonal_periods in para['seasonal_periods']
                )
        r = pd.DataFrame(r).sort_values('mean_rmse', ascending=True).reset_index(drop=True)
        self.para_search_records = r
    
    def fitcv(self, para, n_jobs=-2, n_spilit=5):
        self.ts_cv = n_spilit
        y = self.y
        self.search_para(para, n_jobs)
        para_ = self.para_search_records[:1]
        y = y.interpolate().dropna()
        model = ETSModel(
            endog=y, 
            error=para_['error'][0], 
            trend=para_['trend'][0], 
            seasonal=para_['seasonal'][0], 
            seasonal_periods=para_['seasonal_periods'][0]).fit(full_output=False, disp=False)
        self.model = model
    
    def predict(self, horizon, sig = 0.5 , CI = False):
        if CI == False:
            pred = np.array(self.model.forecast(horizon))
            return pred
        if CI == True:
            self.alpha = sig 
            z = norm.ppf(1-sig/2)
            SE = y.std() * np.sqrt(1 + 1/np.arange(1, horizon+1))
            forecast_mean = self.model.forecast(horizon)
            forecast_lo = forecast_mean - z * SE, 
            forecast_hi = forecast_mean + z * SE
            pred = pd.DataFrame({
                'forecast_mean': np.array(forecast_mean),
                'forecast_lo': np.array(forecast_lo[0]),
                'forecast_hi': np.array(forecast_hi)
            })
            return pred
            
