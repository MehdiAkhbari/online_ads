import numpy as np
# import modin.pandas as pd
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
import config

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
    return X



def calc_tes(data, user_visit_no, ranks_list):
    """
    This function calculates the treatment effects for the ads with ranks in "rank_list" for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
    The output is saved in columns te_1, ..., te_{max_adv_rank} of the dataframe "data"
    """
    start_time = time.perf_counter()
    for rank in ranks_list:
        X = construct_X(data, user_visit_no=user_visit_no, ad_rank=rank)
        var_name = f"te_{rank}"
        exec(f"data.loc[data['user_visit_no'] == user_visit_no, 'temp'] = config.cf_{rank}.const_marginal_effect(X[data['user_visit_no'] == user_visit_no])")
        data.loc[data['user_visit_no'] == user_visit_no, var_name] = data.loc[data['user_visit_no'] == user_visit_no, 'temp']
        # if rank % 10 == 1:
        #     print(f"rank {rank} done!")
    data = data.drop(['temp'], axis=1)
    finish_time = time.perf_counter()
    print(f"finished calculating te's for rank {rank} in {finish_time - start_time} seconds")


def calc_base_ad_ctr(data, user_visit_no):
    """
    This function calculates E(y0|X=x) for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
    The output is saved in columns y_{base_ad} of the dataframe "data"
    """
    start_time = time.perf_counter()
    # Define X variables (Note that I am not using previous_clicks and i mpression_repeat variables here, because I'm only using base ad repeats and clicks here)
    X = data[['previous_clicks_all_ads',
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
    var_name = f"y_{base_ad}"
    data.loc[data['user_visit_no'] == user_visit_no, var_name] = config.base_ad_y_model.predict(X[data['user_visit_no'] == user_visit_no])
    finish_time = time.perf_counter()
    # print(f"finished calculating y0 in {finish_time - start_time} seconds")


def calc_ctrs(data, user_visit_no):
    """
    This function calculates the click rates of all ads for the subset of DataFrame "data" for which the "user_visit_no" is a specific number by adding y_{base_ad} and treatment effects.
    The output is saved in columns y_1, ..., y_{max_adv_rank} of the dataframe "data"
    """
    start_time = time.perf_counter()
    for rank in config.ranks_list:
        y_var_name = f'y_{rank}'
        te_var_name = f'te_{rank}'
        y_base_ad = f'y_{base_ad}'
        data.loc[data['user_visit_no'] == user_visit_no, y_var_name] = data.loc[data['user_visit_no'] == user_visit_no, te_var_name] + data.loc[data['user_visit_no'] == user_visit_no, y_base_ad]
        # set y_{rank} to 0 if it is negative
        data.loc[data['user_visit_no'] == user_visit_no, y_var_name] = data.loc[data['user_visit_no'] == user_visit_no, y_var_name].apply(lambda x: max(x, 0))
    finish_time = time.perf_counter()
    # print(f"finished calculating y_i's in {finish_time - start_time} seconds")


def create_chosen_ad_vars(data):
    """
    This functions initializes two sets of variable in the dataframe "data":
    1) chosen_ad_{ad}: shows the rank of the the top {ad} chosen ad, ex: chosen_ad_1 is the rank of the top ad chosen to be shown
    2)chosen_ad_y_{ad}: shows the corresponding treatment effect of that ad
    Initially, all these columns are NaN
    3) num_ads:  number of ads to be shown (currently nan)

    Inputs:
    - data: the dataframe

    """
    for ad in range(1, config.max_ads_per_page + 1):
        var_name1 = f"chosen_ad_{ad}"
        data.loc[:, var_name1] = np.nan


    for ad in range(1, config.max_ads_per_page + 1):
        var_name2 = f"chosen_ad_y_{ad}"
        data.loc[:, var_name2] = np.nan


    for ad in range(1, config.max_ads_per_page + 1):
        var_name2 = f"chosen_ad_click_dummy_{ad}"
        data.loc[:, var_name2] = np.nan
    data.loc[:, 'num_ads'] = np.nan



def find_optimal_ads(row, y_cols):
    """
    This functions calculates optimal ads (based on highest treatment effects) to be shown to the impression in each row. based on the calculated treatment effects y_i s
    Inputs: 
        - row: the row of the dataframe that it is applied to
        it has to include indices y_cols and "ads_on_page" (determines how many ads to choose)
    
    Returns: 
        - chosen_ads: a list of ads to be shown
        - chosen_ad_ys: a list of the corresponding treatment effects
    """


    # sort the values by the value of the criteria
    sorted_ads = row[y_cols].sort_values(ascending=False).index.to_list()
    l = min(row['ads_on_page'], config.max_ads_per_page)    # number of ads to be shown on each visit
    chosen_ads = sorted_ads[0 : l]
    # creates a list of chosen ad ranks
    chosen_ads = [int(element.strip("y_")) for element in chosen_ads]
    chosen_ad_ys = row[y_cols].sort_values(ascending=False).values[0:l]
    return chosen_ads, chosen_ad_ys



def create_chosen_ad_columns(data, user_visit_no):
    """
    This function finds the optimal ads for the subsection of "data" for which user_visit_no == user_visit_no
    The chosen ads and their corresponding click rates are saved in 'chosen_ad_{ad}' and 'chosen_ad_y_{ad}'
    """
    # select treatment effect columns
    # te_cols = data.loc[0: 1, :].filter(regex="^te_", axis=1).columns
    # select ctr columns:
    y_cols = data.loc[0: 1, :].filter(regex="^y_", axis=1).columns


    for index, row in data[data['user_visit_no'] == user_visit_no].iterrows():
        
        chosen_ads, chosen_ad_ys = find_optimal_ads(row, y_cols)
        chosen_ads = [int(element) for element in chosen_ads]
        l = len(chosen_ads)
        last_chosen_ad_name = f"chosen_ad_{l}"
        # last_chosen_ad_te_name = f"chosen_ad_te_{l}"
        last_chosen_ad_y_name = f"chosen_ad_y_{l}"
        data.loc[index, 'chosen_ad_1': last_chosen_ad_name] = chosen_ads
        data.loc[index, 'chosen_ad_y_1' : last_chosen_ad_y_name] = chosen_ad_ys
        data.at[index, 'num_ads'] = int(l)
        # if index % 10000 == 0:
        #     print(f"index {index} done!")



def update_repeats(data, user_visit_no):
    """
    This function updates the number of previous impression on data after user visit number user_visit_no.
    For example, after a user visits a page for the first time, and observes optimal ads (say ads 2, 5, 10), the initial impressions for all subsequent visits of that user, the number of previous impressions on ads 2, 5, 10 increases by 1. 
    """
    for index, row in data[data['user_visit_no'] == user_visit_no].iterrows():
        for chosen_ad_no in range(1, int(row['num_ads']) + 1):
            var_name = f"chosen_ad_{chosen_ad_no}"
            chosen_ad = int(row[var_name])
            col_name = f'r_{chosen_ad}'
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name] = row[col_name] + 1
        # if index % 10000 == 0:
        #     print(f"index {index} done!")


def update_clicks(data, user_visit_no):
    """
    This function updates the number of previous clicks on data after user visit number user_visit_no.
    For example, after a user visits a page for the first time, and clicks on ad 5, c_5 increases by 1 for all subsequent user impressions. 
    It also updates the column "previous_clicks_all_ads"
    """

    for index, row in data[data['user_visit_no'] == user_visit_no].iterrows():
        total_clicks_on_impression = 0
        for chosen_ad_no in range(1, int(row['num_ads']) + 1):
            var_name = f"chosen_ad_{chosen_ad_no}"
            chosen_ad = int(row[var_name])
            ctr_var = f'y_{chosen_ad}'
            col_name = f'c_{chosen_ad}' # the column name to be updated (if ad 5 is clicked on, c_5 will increase by 1 for all subsequent impressions)
            click_dummy_var =f'chosen_ad_click_dummy_{chosen_ad_no}'
            rand_click = np.random.rand()   # a random number simulating user's click. User will click if rand_click < y_{chosen_ad}
            data.loc[index, click_dummy_var] = int(rand_click <= row[ctr_var])
            total_clicks_on_impression = data.loc[index, click_dummy_var]
            
            
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name] = int(row[col_name] + data.loc[index, click_dummy_var])
        data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), 'previous_clicks_all_ads'] = int(row['previous_clicks_all_ads'] + total_clicks_on_impression)
        # if index % 10000 == 0:
        #     print(f"index {index} done!")    



