import pandas as pd
import numpy as np
from tf_toolbox.transformer import MinMaxNormalization

class MICE:
    def __init__(self, df):
        self.df = df
    
    def impute(
            self, estimator, first_fill='mean', 
            scaling=False, iterations=10, 
            return_type='data', random_state=1):
        self.first_fill = first_fill
        self.return_type = return_type
        self.random_state = random_state
        self.iterations = iterations
        self.scaling = scaling
        self.cate_cols = self.df.select_dtypes('O').columns

        if len(self.cate_cols) > 0:
            self.scaling = True
            print('Impute categorical requires scaling.')

            num_cols = self.df.select_dtypes([float, int]).columns
            self.scaler = MinMaxNormalization(X=self.df[num_cols])
            self.scaler.fit()
            self.scaled_df = self.df.copy()
            self.scaled_df[num_cols] = self.scaler.transform(self.df[num_cols])

            self.pre_imputed_df = create_dummies_with_missing_columns(df=self.scaled_df)
            self.imputed_df = mice(
                df=self.pre_imputed_df, 
                estimator=estimator, 
                iterations=self.iterations,
                return_type=self.return_type)
            
            self.post_imputed_df=adjust_imputed_dummies(
                imputed_df=self.imputed_df, 
                pre_imputed_df=self.pre_imputed_df, 
                cate_cols=self.cate_cols)
            
            self.dummy_cols = get_dummy_colnames(
                df=self.post_imputed_df, 
                cate_cols=self.cate_cols)
            
            self.df_output = inverse_dummies(
                post_imputed_df=self.post_imputed_df, 
                dummy_cols=self.dummy_cols)
            
        else:
            if scaling == False:
                self.df_output = mice(
                    df=self.df, 
                    estimator=estimator, 
                    iterations=self.iterations,
                    return_type=self.return_type)
            elif scaling == True:
                self.scaling = True
                self.scaler = MinMaxNormalization(X=self.df[num_cols])
                self.scaler.fit()
                self.scaled_df = self.df.copy()
                self.scaled_df[num_cols] = self.scaler.transform(self.df[num_cols])

                self.df_output = mice(
                    df=self.scaled_df, 
                    estimator=estimator, 
                    iterations=self.iterations,
                    return_type=self.return_type)
        if self.scaling == True:
            self.df_output[num_cols] = self.scaler.back_transform(self.df_output[num_cols])
        return self.df_output
    
def mice(df, estimator, first_fill='mean', iterations=10, return_type='data', random_state=1):
    """
    df: pandas dataframe
    estimator: preditecive model used to impute data
    first_fill: the inital imputation methods, must of one of "mean" or "median"
    iterations: num of iterations 
    return_type: "data" or "data+loss"

    """
    if first_fill == 'mean':
        df0 = df.fillna(df.mean()).copy()
        df_ = df.fillna(df.mean()).copy()
    elif first_fill == 'median':
        df0 = df.fillna(df.median()).copy()
        df_ = df.fillna(df.median()).copy()
    for __ in range(iterations):
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                col_missing_idx = df[df[col].isnull()].index.values
                subset = df_.iloc[df.index.difference(col_missing_idx)]
                X_train, y_train = subset.drop(columns=col), subset[col]
                X_test = df_.drop(columns=col).iloc[col_missing_idx]
                try:
                    imputer = estimator(random_state=random_state)
                except:
                    imputer = estimator()
                imputer.fit(X=np.array(X_train), y=np.array(y_train))
                y_pred = imputer.predict(X_test)
                df_[col].loc[col_missing_idx] = y_pred
        loss = df0 - df_
    if return_type == 'data':
        return df_
    elif return_type == 'data+loss':
        return df_, loss
    
def create_dummies_with_missing_columns(df):
    """
    This function creates dummies variable for data frame with missing categorical variables.
    """
    df_cp = df.copy()
    cate_columns = df_cp.select_dtypes('object').columns.values
    missing_columns = df_cp.isnull().sum()[df.isnull().sum() > 0].index.values
    missing_cate_columns = [i for i in cate_columns if i in missing_columns]

    df_dummies = pd.get_dummies(df_cp, drop_first=False)
    for col in missing_cate_columns:
        dummy_cols = df_dummies.filter(regex=col+'_').columns.values
        print(dummy_cols)
        for dummy_col in dummy_cols:
            df_dummies[dummy_col] = np.where(df_cp[col].isnull()==True, np.nan, df_dummies[dummy_col])

    return df_dummies

def adjust_imputed_dummies(imputed_df, pre_imputed_df, cate_cols):
    """
    This function cleans up the inputed dummies to make sure one categorical
    outcome is allowed.
    """
    imputed_df_cp = imputed_df.copy()
    for col in cate_cols:
        for i in imputed_df_cp.filter(regex=col+'_').columns.values:
            imputed_df_cp[i] = np.where(
                pre_imputed_df[i].isnull() == True, 
                np.where(
                    imputed_df_cp[i] >= imputed_df_cp.filter(regex='d'+'_').max(axis=1),
                    1, 0
                ), 
                imputed_df_cp[i]
            )
    return imputed_df_cp

def inverse_dummies(post_imputed_df, dummy_cols):
    """
    This function inverse dummies back to categorical data form.
    """
    df_cate = pd.from_dummies(post_imputed_df[dummy_cols], sep='_')
    final_df = pd.concat([
        post_imputed_df.drop(columns=dummy_cols),
        df_cate], axis='columns')
    return final_df 

def get_dummy_colnames(df, cate_cols):
    """
    This function find column that created from dummies.
    """
    output = []
    df_col = df.columns
    for i in cate_cols:
        for j in df_col:
            if i+'_' in j:
                output.append(j)
    return output
