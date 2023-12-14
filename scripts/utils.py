import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator
from econml.sklearn_extensions.model_selection import GridSearchCVList
import time
import joblib

# set number of cores to use
n_jobs = 30

def read_data(data_set_name):
    file_name = f"..\\data\\{data_set_name}"
    data = pd.read_stata(file_name)
    return data


def prepare_data(data, base_ad=50, max_ad=100):
    # Set base treatment advertiser_rank =0
    data.loc[data['advertiser_rank'] == base_ad, 'advertiser_rank'] = 0
    # set advertiser_rank = 101 for 100+ ranked advertisers
    data.loc[data['advertiser_rank'] > max_ad, 'advertiser_rank'] = (max_ad + 1)

def extract_ranks(data):
    # extract the list of available advertiser_ranks
    ranks_list = data['advertiser_rank'].value_counts().sort_index().index.tolist()
    return ranks_list


class PropensityModel(BaseEstimator):
    def __init__(self):
        self.lr = LogisticRegression(max_iter=2000)

    def predict_proba(self, X, X_indices=slice(-27,-1)):
        return self.lr.predict_proba(X[:,X_indices])


    
    # X_indices are the ones that are used for the estimation of the propensity score
    def fit(self, X, y, X_indices=slice(-27,-1)):
        self.lr.fit(X[:,X_indices], y)
        return self

 # Instantiate propensity_model from the PropensityModel class
propensity_model = PropensityModel()


# Define the hyperparameters to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [1000 , 2000, 3000, 5000]
}

# Define the hyperparameters to search over
cf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [1000 , 2000, 3000, 5000],
    'max_samples': [0.1, 0.2, 0.3]
}




def define_xyt(data):
    # define X
    X = data.drop(['publisher_subject', 'advertiser_rank', 'is_clicked', 'event_no', 'prop'], axis=1)
    
    # define T
    T = data['advertiser_rank']
    
    # define Y
    Y = data['is_clicked']
    return X, Y, T


def m_model_best_estimator(X, Y, n_jobs=30):
    start_time = time.perf_counter()
    m_model = RandomForestRegressor(verbose=0, n_jobs=n_jobs)
     
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=m_model, param_grid=param_grid, cv=5)
    grid_search.fit(X, Y)
    
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    finish_time = time.perf_counter()
    print(f"finished tuning the M model in {finish_time - start_time} seconds")
    return best_params


    

# def causal_forest_estimate(X, Y, T, cf_param_grid):
#     # tune the model:
#     start_time = time.perf_counter()

#     cf.tune(
#                 Y=Y,
#                 T=T,
#                 X=X,
#                 params=cf_param_grid)
    
#     finish_time = time.perf_counter()
#     print(f"finished tuning the model in {finish_time - start_time} seconds")

#     # fit the model using tuned parameters:
#     start_time = time.perf_counter()
    
#     cf.fit(Y=Y, T=T, X=X, inference="blb", cache_values=True)
    
#     finish_time = time.perf_counter()
#     print(f"finished fitting the model in {finish_time - start_time} seconds")
#     return cf



    




