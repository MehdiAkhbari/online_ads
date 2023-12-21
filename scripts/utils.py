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
import pickle

# set number of cores to use
n_jobs = 36
base_ad = 50



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


# define the Custom Treatment Model class (its exactly the PropensityModel above. However, I had to include it because I ran the initial model with this one):
class CustomTreatmentModel(BaseEstimator):
    def __init__(self):
        self.lr = LogisticRegression(max_iter=2000)

    def predict_proba(self, X, X_indices=slice(-27,-1)):
        return self.lr.predict_proba(X[:,X_indices])


    
    # X_indices are the ones that are used for the estimation of the propensity score
    def fit(self, X, y, X_indices=slice(-27,-1)):
        self.lr.fit(X[:,X_indices], y)
        return self







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

  
    
###########################Simulation Utility Fuctions########################

def construct_X(data, user_visit_no, ad_rank):
    """ 
    This function updates the inputs for estimation so the estimates are for all user visits with a specific user_visit_no, and a specific ad_rank.
    After calling this function, you can estimate the treatment effect for ad ad_rank and the subset of data for which user_visit_no = user_visit_no.
    """
    # Define X variables
    X = data[['impression_repeat', 'previous_clicks', 'previous_clicks_all_ads',
        'impression_repeat_base_ad', 'previous_clicks_base_ad', 'total_visits',
        'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5', 'visit_s6',
        'visit_s7', 'visit_s8', 'visit_s9', 'visit_s10', 'visit_s11',
        'visit_s12', 'visit_s13', 'visit_s14', 'visit_s15', 'visit_s16',
        'visit_s17', 'visit_s18', 'visit_s19', 'visit_s20', 'visit_s21',
        'visit_s22', 'visit_s23', 'visit_s24', 'visit_s25', 'visit_s26',
        'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8',
        'sub_9', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15',
        'sub_16', 'sub_17', 'sub_18', 'sub_19', 'sub_20', 'sub_21', 'sub_22',
        'sub_23', 'sub_24', 'sub_25', 'sub_26', 'mobile']]

    # Construct X variable for the input to the causal forest
    # a) construct base ad initial clicks and repeats

    base_ad_str = f"r_{base_ad}"
    X.loc[data['user_visit_no'] == 1, 'impression_repeat_base_ad'] = data[data['user_visit_no'] == 1][base_ad_str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)

    base_ad_str = f"c_{base_ad}"
    X.loc[data['user_visit_no'] == 1, 'previous_clicks_base_ad'] = data[data['user_visit_no'] == 1][base_ad_str]

# b) construct each ad's initial clicks and repeats
    str = f"r_{ad_rank}"
    X.loc[data['user_visit_no'] == user_visit_no, 'impression_repeat'] = data[data['user_visit_no'] == user_visit_no][str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)
    str = f"c_{ad_rank}"
    X.loc[data['user_visit_no'] == user_visit_no, 'previous_clicks'] = data[data['user_visit_no'] == user_visit_no][str]




def calc_tes(data, user_visit_no, ranks_list):
    """
    This function calculates the treatment effects for the ads with ranks in "rank_list" for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
    The output is saved in columns te_1, ..., te_{max_adv_rank} of the dataframe "data"
    """
    for rank in ranks_list:
        construct_X(data, user_visit_no=user_visit_no, ad_rank=rank)
        var_name = f"te_{rank}"
        exec(f"data.loc[data['user_visit_no'] == user_visit_no, 'temp'] = cf_{rank}.const_marginal_effect(X[data['user_visit_no'] == user_visit_no])")
        data.loc[data['user_visit_no'] == user_visit_no, var_name] = data.loc[data['user_visit_no'] == user_visit_no, 'temp']
    data = data.drop(['temp'], axis=1)


def create_chosen_ad_vars(data, max_ads_per_page):
    """
    This functions initializes two sets of variable in the dataframe "data":
    1) chosen_ad_{ad}: shows the rank of the the top {ad} chosen ad, ex: chosen_ad_1 is the rank of the top ad chosen to be shown
    2)chosen_ad_te_{ad}: shows the corresponding treatment effect of that ad
    Initially, all these columns are NaN

    Inputs:
    - data: the dataframe
    - max_ads_per_page: the maximum number of ads to be shown (= the number of columns created for each set of variables)
    """
    for ad in range(1, max_ads_per_page + 1):
        var_name1 = f"chosen_ad_{ad}"
        data[var_name1] = np.nan


    for ad in range(1, max_ads_per_page + 1):
        var_name2 = f"chosen_ad_te_{ad}"
        data[var_name2] = np.nan



def find_optimal_ads(row):
    """
    This functions calculates optimal ads (based on highest treatment effects) to be shown to the impression in each row. based on the calculated treatment effects te_i s
    Inputs: 
        - row: the row of the dataframe that it is applied to
        it has to include indices te_cols and "ads_on_page" (determines how many ads to choose)
    
    Returns: 
        - chosen_ads: a list of ads to be shown
        - chosen_ad_tes: a list of the corresponding treatment effects
    """
    # sort the values by the value of the criteria
    sorted_ads = row[te_cols].sort_values(ascending=False).index.to_list()
    chosen_ads = sorted_ads[0 : row.ads_on_page]
    # creates a list of chosen ad ranks
    chosen_ads = [int(element.strip("te_")) for element in chosen_ads]
    chosen_ad_tes = row[te_cols].sort_values(ascending=False).values[0:row['ads_on_page']]
    return chosen_ads, chosen_ad_tes



def create_chosen_ad_columns(data, user_visit_no):
    """
    This function find the optimal ads for the subsection of "data" for which user_visit_no == user_visit_no
    The chosen ads and their corresponding TEs are saved in 'chosen_ad_{ad}' and 'chosen_ad_te_{ad}'
    """
    for index, row in data[data['user_visit_no'] == user_visit_no].iterrows():
        print(index)
        # print(data_temp.loc[index, :])
        chosen_ads, chosen_ad_tes = find_optimal_ads(row)
        print(chosen_ads)
        l = min(row['ads_on_page'], max_ads_per_page)
        last_chosen_ad_name = f"chosen_ad_{l}"
        last_chosen_ad_te_name = f"chosen_ad_te_{l}"
        data.loc[index, 'chosen_ad_1': last_chosen_ad_name] = chosen_ads
        data.loc[index, 'chosen_ad_te_1' : last_chosen_ad_te_name] = chosen_ad_tes






