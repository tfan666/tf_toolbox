from __future__ import print_function
from this import d
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.metrics import classification_report, brier_score_loss, log_loss, recall_score, precision_score, accuracy_score
import sys
from prophet import Prophet
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import seaborn as sns
import numpy as np
import pandas as pd

class eda:
    """
    This class stores functions to do some basic EDA analysis
    """
    def __init__(self, df):
        self.df = df

    def plot_distribution(self, col):
        sns.set(rc={'figure.figsize':(12,8)})
        x = self.df[col]
        missing_pct = round(x.isnull().sum()/len(x)*100, 1)
        skewness = round(skew(x), 2)
        _kurtosis = round(kurtosis(x), 2)
        ax = sns.histplot(x = x)
        ax.set_title(
            f"Histrogram for {col} | Skewness: {skewness} | Kurtosis: {_kurtosis} | Missing Pct: {missing_pct}% "
        )
    
    def show_value_count(self, cols, top=10):
        df = self.df
        d  = pd.DataFrame(df[cols].value_counts())
        d = pd.DataFrame({
            'cate': d.index[:top],
            'cnt': d[cols][:top]
        }).reset_index(drop=True)
        missing_cnt = df[cols].isnull().sum()
        other_cnt = len(df) - missing_cnt - len(d)

        d = pd.concat([
            pd.DataFrame({
                'cate': ['(missing)'],
                'cnt': [missing_cnt]
            }),
            pd.DataFrame({
                'cate': ['(other)'],
                'cnt': [other_cnt]
            }),
            d
            ])\
            .sort_values('cnt', ascending=False)\
            .reset_index(drop=True).\
            assign(
                pct = lambda x: round(x.cnt/len(df),2)
                )
        d['cnt'] = d['cnt'].astype(int)

        return d
    
    def missing_report(self):
        df = self.df
        missing_report = df.isnull().sum()/len(df)
        return missing_report

    def explore_cate(self, keyword_filter=None):
        df = self.df
        if keyword_filter == None:
            interact(
                self.show_value_count, 
                cols = df.select_dtypes('object').columns)
        else:
            keywords = df.select_dtypes('object').columns.str.contains(keyword_filter)
            interact(
                self.show_value_count, 
                cols = df.select_dtypes('object').columns[keywords])

    def explore_univar_dist(self, keyword_filter=None):
        df = self.df
        if keyword_filter == None:
            interact(
                self.plot_distribution, 
                col = df.select_dtypes(['float32', 'float64', 'int32', 'int64']).columns)


def f2_score(y_true=None, y_pred=None, precision=None, recall=None, type='direct'):
    """
    Harmonic weighted mean of precision and recall, with
    more weight on recall.
    """
    if type=='direct':
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f2_score = (1+ 2**2) / ((2**2/recall) + 1/precision)
        return f2_score

    if type=='indirect':
        f2_score = (1+ 2**2) / ((2**2/recall) + 1/precision)
        return f2_score
    else:
        sys.exit("type has to be one of 'direct' or 'indirect'.")