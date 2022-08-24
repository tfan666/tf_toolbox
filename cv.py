import numpy as np
import pandas as pd 

def merge_array(a,b):
    """
    Helper function for generate_k_fold_index()
    """
    c = list(a) + list(b)
    return np.array(c)

def generate_k_fold_index(X, y, k, shuffle=True, stratified=False, random_state=1):
    """
    Create k-fold index. 
    """
    assert len(X) == len(y)
    np.random.seed(random_state)
    cv_idx = X.index.values.copy()
    cv_dict = {'k':[], 'idx':[]}
    if shuffle == True:
        np.random.shuffle(cv_idx)
    if stratified == False:
        dist = len(cv_idx) // k
        for i in range(k-1):
            cv_dict['k'].append(i)
            cv_dict['idx'].append(cv_idx[i*dist: (i+1)*dist])
        cv_dict['k'].append(k-1)
        cv_dict['idx'].append(cv_idx[(k-1)*dist:])
    else:
        y = pd.Series(y)
        groups = y.unique()
        for group in groups:
            y_sub = y[y==group]
            cv_idx = y_sub.index.values.copy()
            dist = len(cv_idx) // k
            
            if len(cv_dict['k'])==0:
                for i in range(k-1):
                    cv_dict['k'].append(i)
                    cv_dict['idx'].append(cv_idx[i*dist: (i+1)*dist])
                cv_dict['k'].append(k-1)
                cv_dict['idx'].append(cv_idx[(k-1)*dist:])
            else:
                for i in range(k-1):
                    cv_dict['idx'][i] = merge_array(cv_idx[i*dist: (i+1)*dist], cv_dict['idx'][i])
                cv_dict['idx'][i] = merge_array(cv_idx[(k-1)*dist:], cv_dict['idx'][i])
                    

    return cv_dict

def random_resample(X, y, target_cnt=None, random_state=1):
    df = pd.concat([X,y], axis=1)
    if target_cnt == None:
        target_cnt = int(y.value_counts().mean())
        y_class = y.unique()
    resampled_df = pd.DataFrame([])
    for i in y_class:
        df_sub = df.loc[y[y==i].index].sample(n=target_cnt, replace=True, random_state=random_state)
        resampled_df = pd.concat([resampled_df, df_sub], axis=0)
    resampled_df = resampled_df.reset_index(drop=True)
    return resampled_df[resampled_df.columns[:-1]], resampled_df[resampled_df.columns[-1]]

def get_cv_results(k, X, y, model, eval_func, random_state=1, clf_threshold=.5, stratified=True):
    """
    This method run k-fold cross-validation with resampling class. 
    """
    kfold_idx = generate_k_fold_index(X=X, y=y, k=k, stratified=stratified)
    def _get_cv_results(i, kfold_idx=kfold_idx, X=X, y=y, model=model, eval_func=eval_func, random_state=random_state, clf_threshold=clf_threshold):
        k_idx = kfold_idx['idx'][i]
        _k_idx = np.array(list(set(y.index.values) - set(k_idx)))
        k_X_train, k_y_train, k_X_test, k_y_test = X.iloc[_k_idx], y[_k_idx], X.iloc[k_idx], y[k_idx]
        rs_k_X_train, rs_k_y_train = random_resample(X=k_X_train, y=k_y_train, target_cnt=None, random_state=1)

        model.fit(X=rs_k_X_train, y=rs_k_y_train)

        k_y_pred = model.predict_proba(k_X_test)[::,1]

        score = eval_func(y_true=k_y_test, y_pred=np.where(k_y_pred>clf_threshold,1,0))

        eval_report = {
            'k': i,
            'score': score
        }
        return eval_report

    r = Parallel(n_jobs=-2)(delayed(_get_cv_results) (
        i, kfold_idx=kfold_idx, X=X, y=y, model=model, eval_func=f2_score) 
        for i in range(k)
        )

    cv_report = pd.DataFrame(r)

    return cv_report