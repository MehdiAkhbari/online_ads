from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, make_scorer, f1_score
import pandas as pd
import numpy as np


X_treat_indices = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 
               'sub_6', 'sub_7', 'sub_8','sub_9', 'sub_10', 
               'sub_11', 'sub_12', 'sub_13',
               'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']

X_treat_indices_nums = range(19, 37)

class PropensityModel(BaseEstimator):
    def __init__(self, **kwargs):
        self.lr = RandomForestClassifier(**kwargs)


    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            return self.lr.predict_proba(X.loc[:, X_treat_indices])
        elif isinstance(X, np.ndarray):
            
            return self.lr.predict_proba(X[:, X_treat_indices_nums])
        else:
            raise TypeError("Input must be a NumPy array or a pandas DataFrame")
        


    def fit(self, X, T):
        if isinstance(X, pd.DataFrame):
            self.lr.fit(X.loc[:, X_treat_indices], T)
        elif isinstance(X, np.ndarray):
            self.lr.fit(X[:, X_treat_indices_nums], T)
        else:
            raise TypeError("Input must be a NumPy array or a pandas DataFrame")
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return self.lr.predict(X.loc[:, X_treat_indices])
        elif isinstance(X, np.ndarray):
            return self.lr.predict(X[:, X_treat_indices_nums])
        else:
            raise TypeError("Input must be a NumPy array or a pandas DataFrame")


    def score(self, X, T):
        T_pred = self.predict(X)
        # return -log_loss(T, T_pred_proba)
        return f1_score(T, T_pred)


    def get_params(self, deep=True):
        return self.lr.get_params(deep)

    def set_params(self, **params):
        self.lr.set_params(**params)
        return self



