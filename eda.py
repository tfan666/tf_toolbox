from __future__ import print_function
from this import d
from ipywidgets import interact, interactive, fixed, interact_manual
import sys
from scipy.stats import skew, kurtosis, shapiro
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


def shapiro_test(x):
    """
    Run Shapiro-Wilk Normality Test and return p value.
    - Null Hypothesis: observed data selected from a normal distribution.
    - Alternative Hypothesis: observed data not selected from a normal distribution.
    """
    p_value = shapiro(x)[1]
    return p_value

def plot_wordcloud(X: pd.Series, fig_size=(12,8), title=''):
    """
    Plot frequency-based on word cloud from pandas df
    """

    wc = WordCloud(
        width = fig_size[0]*100, 
        height = fig_size[1]*100,
        background_color ='white',
        include_numbers=True,
        relative_scaling=1,
        min_font_size = 10)

    p = wc.generate_from_frequencies(dict(X.value_counts()))

    plt.figure(figsize = fig_size, facecolor = None)
    plt.imshow(p)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.title(title, fontsize=fig_size[0])
    plt.show()
    

class eda:
    """
    This class stores functions to do some basic EDA analysis
    """
    def __init__(self, df):
        self.df = df

    def size(self):
        n_cols = self.df.shape[1]
        n_rows = self.df.shape[0]
        size = self.df.memory_usage().sum()/1e9
        print(f"This dataset has {n_cols} columns and {n_rows} rows. The estimated storage size is {size} GB ")
        
    def distinct_report(self):
        distinc = pd.DataFrame({
            'column': self.df.columns,
            'distinct_cnt': [len(self.df[col].unique()) for col in self.df.columns] 
            }).assign(
            distinct_ratio = lambda x: (x.distinct_cnt / len(self.df)).round(3)
            )
        return distinc

    def numeric_eda_report(self):
        dict_report = self.distinct_report()
        distri_report = self.distribution_report()
        mis_report = self.missing_data_report()
        norm_report = self.normality_test()
        num_eda_report = pd.merge(
                pd.merge(
                    pd.merge(
                        mis_report, dict_report, on='column'), 
                    distri_report.drop(columns='count'), on='column'),
                norm_report.drop(columns='p_value'), on='column')
        
        return num_eda_report

    def categorical_eda_report(self):
        dict_report = self.distinct_report()
        mis_report = self.missing_data_report()
        mode_report = pd.DataFrame({
            'column': self.df.select_dtypes('O').columns,
            'most_freqent_value': [self.df[col].value_counts().index.values[0] for col in self.df.select_dtypes('O').columns],
            'most_freqent_value_cnt': [self.df[col].value_counts().values[0] for col in self.df.select_dtypes('O').columns]
        }).assign(
            most_freqent_ratio = lambda x: (x.most_freqent_value_cnt / len(self.df)).round(3)
        )
        categorical_eda_report = pd.merge(
            mis_report,
            pd.merge(
                dict_report, 
                mode_report, 
                on='column'), 
            on='column')
        return categorical_eda_report

    
    def column_type_report(self):
        """
        Compute table format column data type report. 
        """
        df = self.df
        
        int_cnt = len(df.select_dtypes(int).columns)
        num_cnt = len(df.select_dtypes(float).columns)
        str_cnt = len(df.select_dtypes('object').columns)
        dt_cnt = len(df.select_dtypes('datetime').columns)

        report = pd.DataFrame({
            'Column Type': ['Numeric', 'Integer', 'String', 'Datetime', 'Total'],
            'Count': [num_cnt, int_cnt, str_cnt, dt_cnt, num_cnt + int_cnt + str_cnt + dt_cnt,]
        })
        return report

    def column_type_plot(self, fig_size = (8,4), fig_title=''):
        sns.set(rc={'figure.figsize':fig_size})
        report = self.column_type_report()
        report = report[report['Column Type'] != 'Total']
        ax = sns.barplot(data=report, x='Column Type', y='Count')
        ax.set_title(fig_title)
        ax.bar_label(ax.containers[0])


    def distribution_report(self, round_digits=2):
        """
        Compute table format distribution-related statistics for numeric and integer columns.

        """
        df = self.df
        dis_table = df.describe().T\
            .reset_index()\
            .rename(columns={'index': 'column'})

        skew_table = pd.DataFrame({
            'column': df.select_dtypes([int, float]).skew().index,
            'skewness': df.select_dtypes([int, float]).skew().values
        })

        kurtosis_table = pd.DataFrame({
            'column': df.select_dtypes([int, float]).kurtosis().index,
            'kurtosis': df.select_dtypes([int, float]).kurtosis().values
        })

        dis_table = pd.merge(
            pd.merge(
                dis_table, skew_table, 
                how='left', on='column'
            ),
            kurtosis_table, 
            how='left', on='column').round(round_digits)
        return dis_table

    def plot_univariate_distribution(self, col, type='histogram', group=None, fig_size=(12,8)):
        """"
        Compute Univariate Histrogram.
        
        """
        sns.set(rc={'figure.figsize':fig_size})
        df = self.df
        x = df[col]
        missing_pct = round(x.isnull().sum()/len(x)*100, 1)
        skewness = round(skew(x.dropna()), 2)
        _kurtosis = round(kurtosis(x.dropna()), 2)
        if type == 'histogram':
            if group ==None:
                ax = sns.histplot(x=x)
            else:
                ax = sns.histplot(x=col, hue=group, data=df)
        elif type == 'density':
            if group ==None:
                ax = sns.kdeplot(x=x)
            else:
                ax = sns.kdeplot(x=col, hue=group, data=df)
        elif type == 'histogram+kde':
            if group ==None:
                ax = sns.histplot(x=x, kde=True)
            else:
                ax = sns.histplot(x=col, hue=group, data=df)
        elif type == 'boxplot':
            if group==None:
                ax = sns.boxplot(y=x)
            else:
                ax = sns.boxplot(y=col, x=group, data=df)
        elif type == 'violin':
            if group==None:
                ax = sns.violinplot(y=x)
            else:
                ax = sns.violinplot(y=col, x=group, data=df)
        else:
            print("Type has to be one of 'box', 'violin', 'density', 'histogram', or 'histogram+kde." )
        if group==None:
            ax.set_title(
                f"Distribution for {col} | Skewness: {skewness} | Kurtosis: {_kurtosis} | Missing Pct: {missing_pct}% ",
                fontsize=fig_size[0]*1.5)
        else:
            ax.set_title(f'Distribution of {col} By {group.capitalize()}', fontsize=fig_size[0]*1.5)


    def plot_bivariate_distribution(self, col, group, type='density', fig_size=(12,8)):
        """
        WIP
        """
        df = self.df
        sns.set(rc={'figure.figsize':fig_size})
        if type == 'histogram':
            ax = sns.histplot(x=col, hue=group, data=df)
        elif type == 'density':
            ax = sns.kdeplot(x=col, hue=group, data=df)
        elif type == 'histogram+kde':
            ax = sns.histplot(x=col,hue=group, kde=True, data=df)
        elif type == 'boxplot':
            ax = sns.boxplot(y=col, x=group, data=df)
        elif type == 'violin':
            ax = sns.violinplot(y=col, x=group, data=df)
        elif type == 'strip':
            ax = sns.stripplot(y=col, x=group, data=df)
        else:
            print("Type has to be one of 'boxplot', 'violin', 'density', 'histogram', 'strip', or 'histogram+kde'." )
        ax.set_title(
            f"Distribution for {col} By {group.capitalize()}",
            fontsize=fig_size[0]
        )

    def plot_grid_univariate_distribution(self, type='histogram', grid=None, group=None, fig_size=(15,12)):
        """
        Plot multiple distribution in grid. Pass 'grid' as tuple (n_rows, n_cols). Default will be
        3 cols. 
        """
    
        df = self.df.select_dtypes([int,float])
        if grid ==None:
            n = len(df.columns)
            n_rows = int(np.ceil(n/3))
            grid = (n_rows, 3)
    
        fig, axs = plt.subplots(
            nrows=grid[0], 
            ncols=grid[1], 
            figsize=fig_size)
        plt.subplots_adjust(hspace=1.5)
        if group==None:
            fig.suptitle("Distribution Plots", fontsize=fig_size[0], y=0.95)
        else:
            fig.suptitle(f"Distribution Plots By {group.capitalize()}", fontsize=fig_size[0], y=0.95)

        # loop through tickers and axes
        for col, ax in zip(df.columns, axs.ravel()):
            skewness = round(skew(df[col].dropna()), 2)
            _kurtosis = round(kurtosis(df[col].dropna()), 2)
            if type == 'histogram':
                if group==None:
                    sns.histplot(x=col, data=df, ax=ax)
                else:
                    sns.histplot(x=col, hue=group, data=self.df, ax=ax)
            elif type == 'density':
                if group==None:
                    sns.kdeplot(x=col, data=df, ax=ax)
                else:
                    sns.kdeplot(x=col, hue=group, data=self.df, ax=ax)
            elif type == 'histogram+density':
                if group==None:
                    sns.histplot(x=col, data=df, kde=True ,ax=ax)
                else:
                    sns.histplot(x=col, hue=group, data=self.df, kde=True ,ax=ax)
            elif type == 'boxplot':
                if group==None:
                    sns.boxplot(y=col, data=df, ax=ax)
                else:
                    sns.boxplot(y=col, x=group, data=self.df, ax=ax)
            elif type == 'violin':
                if group==None:
                    sns.violinplot(y=col, data=df, ax=ax)
                else:
                    sns.violinplot(y=col, x=group, data=self.df, ax=ax)
            else:
                print("Type has to be one of 'boxplot', 'violin', 'density', 'histogram', or 'histogram+kde.")
            if group==None:
                ax.set_title(f"Skewness: {skewness} | Kurtosis: {_kurtosis} ")


        plt.show()
    
    def show_value_count(self, column, top=10):
        df = self.df
        d  = pd.DataFrame(df[column].value_counts())
        d = pd.DataFrame({
            'cate': d.index[:top],
            'cnt': d[column][:top]
        }).reset_index(drop=True)
        missing_cnt = df[column].isnull().sum()
        other_cnt = len(df[column].value_counts()[top:]) - missing_cnt 

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
    
    def univariate_categorical_countplot(self, column, top=10, fig_size=(10,6)):
        sns.set(rc={'figure.figsize':fig_size})
        count_table = self.show_value_count(column=column, top=top)
        ax = sns.barplot(data=count_table, x='cate', y='pct')
        ax.bar_label(ax.containers[0])
        ax.set_title('Countplot For '+column, fontsize=fig_size[0])

    def plot_grid_univariate_countplot(self, grid=None, group=None, top=10, fig_size=(15,12)):
        """
        Plot multiple countplot in grid. Pass 'grid' as tuple (n_rows, n_cols). Default will be
        2 cols. 
        """
        df = self.df.select_dtypes(['object'])
        if grid ==None:
            n = len(df.columns)
            n_rows = int(np.ceil(n/2))
            grid = (n_rows, 2)
    
        fig, axs = plt.subplots(
            nrows=grid[0], 
            ncols=grid[1], 
            figsize=fig_size)
        plt.subplots_adjust(hspace=0.5)
        if group==None:
            fig.suptitle("Countplot Plots", fontsize=fig_size[0], y=0.95)
        else: 
            fig.suptitle(f"Countplot Plots By {group.capitalize()}", fontsize=fig_size[0], y=0.95)

        # loop through tickers and axes
        for col, ax in zip(df.columns, axs.ravel()):
            count_table = self.show_value_count(column=col, top=top)
            if group==None:
                sns.barplot(data=count_table, x='cate', y='pct', ax=ax)
                ax.bar_label(ax.containers[0])
                ax.set_xlabel('')
                ax.set_title(col)
            else:
                sns.countplot(x=col, hue=group ,data=self.df, ax=ax)
                ax.set_xlabel('')
                ax.set_title(f"{col} Count By {group.capitalize()}")
                
        plt.show()

    
    def missing_data_report(self):
        """
        Compute missing data report.

        """
        df = self.df
        missing_report = pd.DataFrame({
            'column': df.columns,
            'row_cnt': len(df),
            'missing_cnt': df.isnull().sum().values,
            'missing_ratio': (df.isnull().sum().values/len(df)).round(3)
        })
        return missing_report

    def missing_data_plot(self, fig_size=(20,20), fig_title=''):
        df = self.df
        sns.set(rc = {'figure.figsize': fig_size})
        ax = sns.heatmap(
            data=np.where(df.isnull() == True,1,0),
            cmap=['#7FB3D5', '#D7DBDD'],
            cbar=False,
            xticklabels=df.columns)
        ax.set_title(fig_title, fontsize=fig_size[0])

    def explore_cate(self, keyword_filter=None):
        """
        Compute Categorical Count Table.
        """
        df = self.df
        if keyword_filter == None:
            interact(
                self.show_value_count, 
                column = df.select_dtypes('object').columns)
        else:
            keywords = df.select_dtypes('object').columns.str.contains(keyword_filter)
            interact(
                self.show_value_count, 
                column = df.select_dtypes('object').columns[keywords])

    def explore_univar_dist(self, type= 'histogram' ,keyword_filter=None):
        """
        Compute Interactive Univariate Distribution Plot. WIP
        """
        df = self.df
        def g(col, type):
            skewness = round(skew(self.df[col].dropna()), 2)
            _kurtosis = round(kurtosis(self.df[col].dropna()), 2)
            missing_pct = round(df[col].isnull().sum()/len(self.df),2)
            if type=='histogram':
                ax = sns.histplot(data=self.df, x=col)
            elif type == 'density':
                ax = sns.kdeplot(data=self.df, x=col)
            elif type == 'histogram+density':
                ax = sns.histplot(data=self.df, x=col, kde=True)
            elif type == 'boxplot':
                ax = sns.boxplot(data=self.df, y=col)
            elif type == 'violin':
                ax = sns.violinplot(data=self.df, y=col)

            ax.set_title(f"Skewness: {skewness} | Kurtosis: {_kurtosis} | Missing: {missing_pct} ")
        if keyword_filter == None:
            interact(
                g, 
                col=self.df.select_dtypes([float, int]).columns, 
                type=type)

    def correlation_plot(self, correlation_method='pearson', fig_size=(20,15), fig_title=None):
        """
        Compute correlation matrix plot. Available correlation methods:
            - Pearson 
            - Kendall
            - Spearsman
        """
        df = self.df
        sns.set(rc={
            'figure.figsize':fig_size
        })
        sns.set_theme(style="white")
        # Compute the correlation matrix
        corr = df.corr(method=correlation_method)
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(
            corr, mask=mask, cmap=cmap, 
            vmax=.3, center=0,
            annot=True, fmt='.2f',
            square=True, linewidths=.5, 
            cbar_kws={"shrink": .5})
        if fig_title==None:
            fig_title = f"{correlation_method.capitalize()} Correalation Matrix"
        ax.set_title(fig_title, fontsize=fig_size[0])

    def association_plot(self, columns=None):
        """
        This function plots assocaition scatterplot.
        """
        df = self.df
        if columns==None:
            columns = df.select_dtypes([float, int])
        df = df[columns]
        g = sns.PairGrid(df)
        g.map(sns.scatterplot)

    def normality_test(self, sig=.05):
        df = self.df.select_dtypes([float, int])
        normality_report = pd.DataFrame({
            'column': df.columns,
            'p_value': df.apply(shapiro_test).round(3).values
        }).assign(
            normality = lambda x: np.where(x.p_value > sig, 'Yes', 'No')
        )
        return normality_report
