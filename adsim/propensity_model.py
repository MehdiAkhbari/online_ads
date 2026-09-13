from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
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
        # DML needs a calibrated probability, not a thresholded decision:
        # with arm shares as lopsided as 150-vs-100,000 rows, a
        # thresholded metric (the previous f1_score) will happily select
        # a model that predicts the majority class everywhere. econml
        # calls this internally during cf.tune() regardless of dataset
        # size, and T is binary for the two-arm estimate.py path but
        # multiclass (one class per advertiser rank) for
        # estimate_joint.py; log_loss handles both natively, and
        # labels=self.lr.classes_ keeps it well-defined even if a given
        # scoring batch's T doesn't include every class the model saw
        # during fit.
        proba = self.predict_proba(X)
        return -log_loss(T, proba, labels=self.lr.classes_)


    def get_params(self, deep=True):
        return self.lr.get_params(deep)

    def set_params(self, **params):
        self.lr.set_params(**params)
        return self



