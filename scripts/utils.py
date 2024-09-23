import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, make_scorer, f1_score
from econml.sklearn_extensions.model_selection import GridSearchCVList
import time
import joblib
import pickle
from propensity_model import PropensityModel
import config


# set number of cores to use
n_jobs = 30
base_ad = 50
max_adv_rank = 100
max_adv_rank_fringe = 200
max_visit_no = 100 # max number of page visits by each user


def read_data(data_set_name):
    file_name = f"..\\data\\{data_set_name}"
    data = pd.read_stata(file_name)
    return data


def prepare_data(data, base_ad=50, max_ad=100):
    # Set base treatment advertiser_rank =0
    data.loc[data['advertiser_rank'] == base_ad, 'advertiser_rank'] = 0
    # set advertiser_rank = 101 for 100+ ranked advertisers
    data.loc[(data['advertiser_rank'] > max_adv_rank) & (data['advertiser_rank'] <= max_adv_rank_fringe), 'advertiser_rank'] = (max_adv_rank + 1)
    data.loc[(data['advertiser_rank'] > max_adv_rank_fringe), 'advertiser_rank'] = (max_adv_rank_fringe + 1)


def extract_ranks(data):
    # extract the list of available advertiser_ranks
    ranks_list = data['advertiser_rank'].value_counts().sort_index().index.tolist()


    return ranks_list








# Define the hyperparameters to search over
param_grid = {
    # 'n_estimators': [50, 100, 200],
    'n_estimators': [100],
    'max_depth': [10, 20, 30],
    'min_samples_split': [1000 , 2000, 5000]
}

# Define the hyperparameters to search over
cf_param_grid = {
    # 'n_estimators': [100, 200, 300],
    'n_estimators': [300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [1000 , 2000, 5000],
    # 'max_samples': [0.1, 0.2, 0.3]
}




def define_xyt(data):
    # define X
    X = data[['impression_repeat', 'impression_repeat_base_ad', 
              'previous_clicks', 'previous_clicks_base_ad', 'previous_clicks_all_ads',
               'total_visits', 
               'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5',
               'visit_s6','visit_s7', 'visit_s8', 'visit_s9', 'visit_s10',
               'visit_s11','visit_s12', 'visit_s13',
               'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 
               'sub_6', 'sub_7', 'sub_8','sub_9', 'sub_10', 
               'sub_11', 'sub_12', 'sub_13',
               'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']]
    
    # define T
    T = data['advertiser_rank']
    
    # define Y
    Y = data['is_clicked']
    return X, Y, T


def m_model_best_estimator(X, Y, param_grid, n_jobs=n_jobs):
    start_time = time.perf_counter()
    m_model = RandomForestRegressor(verbose=0, n_jobs=n_jobs)
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=m_model, param_grid=param_grid, cv=5)
    grid_search.fit(X, Y)
    
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    finish_time = time.perf_counter()
    print(f"finished tuning the M model in {finish_time - start_time} seconds")
    return best_params, best_estimator


def e_model_best_estimator(X, T, param_grid, n_jobs=n_jobs):
    start_time = time.perf_counter()
    e_model = PropensityModel()

    # Define a custom scorer using log_loss
    # log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    # Define the scorer
    f1_scorer = make_scorer(f1_score)

    # Perform grid search cross-validation
    # grid_search = GridSearchCV(estimator=e_model, param_grid=param_grid, cv=5, n_jobs=n_jobs, scoring=log_loss_scorer)
    grid_search = GridSearchCV(estimator=e_model, param_grid=param_grid, scoring=f1_scorer, cv=5)
    grid_search.fit(X, T)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    finish_time = time.perf_counter()
    print(f"Finished tuning the E model in {finish_time - start_time} seconds")
    return best_params, best_estimator

    
  
    
###########################Monopoly Simulation Utility Fuctions########################

def construct_X(data, user_visit_no, ad_rank):
    """ 
    This function updates the inputs for estimation so the estimates are for all user visits with a specific user_visit_no, and a specific ad_rank.
    After calling this function, you can estimate the treatment effect for ad ad_rank and the subset of data for which user_visit_no = user_visit_no.
    """
    # Define X variables
    X = data[['impression_repeat', 'impression_repeat_base_ad', 
              'previous_clicks', 'previous_clicks_base_ad', 'previous_clicks_all_ads',
               'total_visits', 
               'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5',
               'visit_s6','visit_s7', 'visit_s8', 'visit_s9', 'visit_s10',
               'visit_s11','visit_s12', 'visit_s13',
               'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 
               'sub_6', 'sub_7', 'sub_8','sub_9', 'sub_10', 
               'sub_11', 'sub_12', 'sub_13',
               'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']]
    
    # Construct X variable for the input to the causal forest
    # a) construct base ad initial clicks and repeats

    base_ad_str = f"r_{base_ad}"
    X.loc[data['user_visit_no'] == user_visit_no, 'impression_repeat_base_ad'] = data[data['user_visit_no'] == user_visit_no][base_ad_str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)

    base_ad_str = f"c_{base_ad}"
    X.loc[data['user_visit_no'] == user_visit_no, 'previous_clicks_base_ad'] = data[data['user_visit_no'] == user_visit_no][base_ad_str]

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


def calc_base_ad_ctr_vector(data, user_visit_no):
    start_time = time.perf_counter()
    X = construct_X(data, user_visit_no=user_visit_no, ad_rank=1) # Here I use cf_1 (the first ads causal forest to estimate y0)
    tau_1 = config.cf_1.const_marginal_effect(X[data['user_visit_no'] == user_visit_no]).reshape(-1, 1)
    m_1 = config.m1.predict(X[data['user_visit_no'] == user_visit_no]).reshape(-1, 1)
    e_1 = config.e1.predict_proba(X[data['user_visit_no'] == user_visit_no]).reshape(-1, 1)
    y_0 = m_1 - tau_1 * e_1 
    finish_time = time.perf_counter()
    print(f"finished calculating base ad ctr in {finish_time - start_time} seconds")
    return y_0

     



def calc_base_ad_ctr(data, user_visit_no):
    """
    This function calculates E(y0|X=x) for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
    The output is saved in columns y_{base_ad} of the dataframe "data"
    """
    start_time = time.perf_counter()
    y_0 = calc_base_ad_ctr_vector(data, user_visit_no)
    var_name = f"y_{base_ad}"
    
    data.loc[data['user_visit_no'] == user_visit_no, var_name] = np.maximum(y_0, 0)


    finish_time = time.perf_counter()
    # print(f"finished calculating y0 in {finish_time - start_time} seconds")


def calc_ctrs(data, vals_data, user_visit_no):
    """
    This function calculates the click rates of all ads for the subset of DataFrame "data" for which the "user_visit_no" is a specific number by adding y_{base_ad} and treatment effects.
    The output is saved in columns y_1, ..., y_{max_adv_rank} of the dataframe "data"
    """
    start_time = time.perf_counter()
    for rank in config.ranks_list:
        y_var_name = f'y_{rank}'
        te_var_name = f'te_{rank}'
        y_base_ad = f'y_{base_ad}'
        rev_var_name = f'rev_{rank}'
        data.loc[data['user_visit_no'] == user_visit_no, y_var_name] = data.loc[data['user_visit_no'] == user_visit_no, te_var_name] + data.loc[data['user_visit_no'] == user_visit_no, y_base_ad]
        # set y_{rank} to 0 if it is negative
        data.loc[data['user_visit_no'] == user_visit_no, y_var_name] = data.loc[data['user_visit_no'] == user_visit_no, y_var_name].apply(lambda x: max(x, 0))
        # revenue = ctr * valuation 
        valuation = vals_data.loc[vals_data['advertiser_rank'] == rank].advertiser_val_cents.to_numpy()[0]
        data.loc[data['user_visit_no'] == user_visit_no, rev_var_name] = data.loc[data['user_visit_no'] == user_visit_no, y_var_name] * valuation
    finish_time = time.perf_counter()
    print(f"finished calculating y_i's in {finish_time - start_time} seconds")


def create_chosen_ad_vars(data):
    """
    This functions initializes three sets of variable in the dataframe "data":
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
        var_name2 = f"chosen_ad_rev_{ad}"
        data.loc[:, var_name2] = np.nan


    for ad in range(1, config.max_ads_per_page + 1):
        var_name2 = f"chosen_ad_click_dummy_{ad}"
        data.loc[:, var_name2] = np.nan
    data.loc[:, 'num_ads'] = np.nan

  



def find_optimal_ads(row, criteria):
    """
    This functions calculates optimal ads (based on highest treatment effects) to be shown to the impression in each row. based on the calculated treatment effects y_i s
    Inputs: 
        - row: the row of the dataframe that it is applied to
        it has to include indices y_cols and "ads_on_page" (determines how many ads to choose)
    
    Returns: 
        - chosen_ads: a list of ads to be shown
        - chosen_ad_ys: a list of the corresponding treatment effects
    """
    # y_cols = data.loc[0: 1, :].filter(regex="^y_", axis=1).columns
    # rev_cols = data.loc[0: 1, :].filter(regex="^rev_", axis=1).columns

    y_cols = row.filter(regex="^y_", axis=0).index
    rev_cols = row.filter(regex="^rev_", axis=0).index


    # sort the values by the value of the criteria
    if criteria == "CTR":
        sorted_ads = row[y_cols].sort_values(ascending=False).index.to_list()
        l = min(row['ads_on_page'], config.max_ads_per_page)    # number of ads to be shown on each visit
        chosen_ads = sorted_ads[0 : l]
        chosen_ads = [int(element.strip("y_")) for element in chosen_ads]

    if criteria == "revenue":
        sorted_ads = row[rev_cols].sort_values(ascending=False).index.to_list() 
        l = min(row['ads_on_page'], config.max_ads_per_page)    # number of ads to be shown on each visit  
        chosen_ads = sorted_ads[0 : l]
        chosen_ads = [int(element.strip("rev_")) for element in chosen_ads]


    # creates a list of chosen ad ranks
    chosen_ad_ys = row[y_cols].sort_values(ascending=False).values[0:l]
    chosen_ad_revs = row[rev_cols].sort_values(ascending=False).values[0:l]
    return chosen_ads, chosen_ad_ys, chosen_ad_revs



def create_chosen_ad_columns(data, user_visit_no, criteria):
    """
    This function finds the optimal ads for the subsection of "data" for which user_visit_no == user_visit_no
    The chosen ads and their corresponding click rates are saved in 'chosen_ad_{ad}' and 'chosen_ad_y_{ad}'
    """
    # select treatment effect columns
    # te_cols = data.loc[0: 1, :].filter(regex="^te_", axis=1).columns
    # select ctr columns:


    for index, row in data[data['user_visit_no'] == user_visit_no].iterrows():
        
        chosen_ads, chosen_ad_ys, chosen_ad_revs = find_optimal_ads(row, criteria)
        chosen_ads = [int(element) for element in chosen_ads]
        l = len(chosen_ads)
        last_chosen_ad_name = f"chosen_ad_{l}"
        # last_chosen_ad_te_name = f"chosen_ad_te_{l}"
        last_chosen_ad_y_name = f"chosen_ad_y_{l}"
        last_chosen_ad_rev_name = f"chosen_ad_rev_{l}"
        data.loc[index, 'chosen_ad_1': last_chosen_ad_name] = chosen_ads
        data.loc[index, 'chosen_ad_y_1' : last_chosen_ad_y_name] = chosen_ad_ys
        data.loc[index, 'chosen_ad_rev_1' : last_chosen_ad_rev_name] = chosen_ad_revs
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
            total_clicks_on_impression += data.loc[index, click_dummy_var]
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name] = int(row[col_name] + data.loc[index, click_dummy_var])
        data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), 'previous_clicks_all_ads'] = int(row['previous_clicks_all_ads'] + total_clicks_on_impression)
        # if index % 10000 == 0:
        #     print(f"index {index} done!")    



def simulate_monopoly(data, vals_data, criteria):
    # file_name = f"data_chunk_{chunk}"
    # create empty columns in the dataframe to fill later
    create_chosen_ad_vars(data)
    # print(f"\n\n\n=======> Chunk #{chunk}")
    start_time = time.perf_counter()


    for i in range(1, max_visit_no + 1):

        start_time_1 = time.perf_counter()
        print(f"\n\n --->Repeat #{i}:")
        # 1) calculate treatment effects, and base ad ctr, then sum them sup and create ctrs for all ads
        # start_time = time.perf_counter()
        calc_tes(data, user_visit_no=i, ranks_list=config.ranks_list)
        calc_base_ad_ctr(data, user_visit_no=i)
        calc_ctrs(data, vals_data, user_visit_no=i)

        # 2) determine what ads are chosen
        # a. create empty columns in the dataframe to fill later
        start_time_2 = time.perf_counter()
        # find the optimal ads and save them and their corresponding ctr's in the dataframe
        create_chosen_ad_columns(data, user_visit_no=i, criteria=criteria)
        finish_time_2 = time.perf_counter()
        print(f"Choosing optimal ads for repeat  {i} finished in {finish_time_2 - start_time_2} seconds!")

    
        # 3) Update repeats and clicks for the next impressions
        # start_time_1 = time.perf_counter()
        start_time_2 = time.perf_counter()
        update_repeats(data, user_visit_no=i)
        finish_time_2 = time.perf_counter()
        print(f"Updating repeats for repeat  {i} finished in {finish_time_2 - start_time_2} seconds!")


        start_time_2 = time.perf_counter()
        update_clicks(data, user_visit_no=i) 
        finish_time_2 = time.perf_counter()
        print(f"Updating clicks for repeat  {i} finished in {finish_time_2 - start_time_2} seconds!")

        finish_time_1 = time.perf_counter()

        print(f"Repeat {i} finished in {finish_time_1 - start_time_1} seconds!")

    finish_time = time.perf_counter()
    print(f"All Repeats finished in {finish_time - start_time} seconds!")
    return data





# def create_actual_ctr_vars(data_s):
#     """
#     This functions initializes a set of variable in the split dataframe "data_s":
#     2)chosen_ad_y_{ad}_actual: shows the ctr of each chosen ad


#     Inputs:
#     - data: the dataframe

#     """
#     for ad in range(1, config.max_ads_per_page + 1):
#         var_name1 = f"chosen_ad_{ad}"
#         data_s.loc[:, var_name1] = np.nan




###########################Duopoly Simulation Utility Fuctions########################



def construct_split_X(data, split_no, user_visit_no, ad_rank):
    """ 
    This function updates the inputs for estimation so the estimates are for all user visits with a specific user_visit_no, and a specific ad_rank.
    After calling this function, you can estimate the treatment effect for ad ad_rank and the subset of data for which user_visit_no = user_visit_no.
    """
    # Define X variables
    
    X = data.loc[data['split'] == split_no, ['impression_repeat_s', 'impression_repeat_base_ad_s', 
              'previous_clicks_s', 'previous_clicks_base_ad_s', 'previous_clicks_all_ads_s',
               'total_visits_s', 
               'visit_s1_s', 'visit_s2_s', 'visit_s3_s', 'visit_s4_s', 'visit_s5_s',
               'visit_s6_s','visit_s7_s', 'visit_s8_s', 'visit_s9_s', 'visit_s10_s',
               'visit_s11_s','visit_s12_s', 'visit_s13_s',
               'sub_1_s', 'sub_2_s', 'sub_3_s', 'sub_4_s', 'sub_5_s', 
               'sub_6_s', 'sub_7_s', 'sub_8_s','sub_9_s', 'sub_10_s', 
               'sub_11_s', 'sub_12_s', 'sub_13_s',
               'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']]
    # remove "_s" from column names in X to be able to run the causal forest model
    X.columns = X.columns.str[:-2]

    # rename var name "mobi" to 'mobile' (revert the above line)
    X = X.rename(columns={'mobi': 'mobile', 'ho': 'hour', 'd': 'day', 'ho': 'hour', 'ads_on_pa': 'ads_on_page'})
    # Construct X variable for the input to the causal forest
    # a) construct base ad initial clicks and repeats

    base_ad_str = f"r_{base_ad}_s"
    X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), 'impression_repeat_base_ad'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][base_ad_str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)

    base_ad_str = f"c_{base_ad}_s"
    X.loc[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no), 'previous_clicks_base_ad'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][base_ad_str]

# b) construct each ad's initial clicks and repeats
    str = f"r_{ad_rank}_s"
    X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), 'impression_repeat'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)
    str = f"c_{ad_rank}_s"
    X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), 'previous_clicks'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][str]
    return X

def construct_actual_X(data, split_no, user_visit_no, ad_rank):
    """ 
    This function updates the inputs for estimation so the estimates are for all user visits with a specific user_visit_no, and a specific ad_rank.
    After calling this function, you can estimate the treatment effect for ad ad_rank and the subset of data for which user_visit_no = user_visit_no.
    """
    # Define X variables
    # X = data.loc[data['split'] ==split_no, ['impression_repeat', 'previous_clicks', 'previous_clicks_all_ads',
    #     'impression_repeat_base_ad', 'previous_clicks_base_ad', 'total_visits',
    #     'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5', 'visit_s6',
    #     'visit_s7', 'visit_s8', 'visit_s9', 'visit_s10', 'visit_s11',
    #     'visit_s12', 'visit_s13', 
    #     'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8',
    #     'sub_9', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'mobile']]

    X = data.loc[data['split'] == split_no, ['impression_repeat', 'impression_repeat_base_ad', 
        'previous_clicks', 'previous_clicks_base_ad', 'previous_clicks_all_ads',
        'total_visits', 
        'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5',
        'visit_s6','visit_s7', 'visit_s8', 'visit_s9', 'visit_s10',
        'visit_s11','visit_s12', 'visit_s13',
        'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 
        'sub_6', 'sub_7', 'sub_8','sub_9', 'sub_10', 
        'sub_11', 'sub_12', 'sub_13',
        'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']]


    base_ad_str = f"r_{base_ad}"
    X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), 'impression_repeat_base_ad'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][base_ad_str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)

    base_ad_str = f"c_{base_ad}"
    X.loc[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no), 'previous_clicks_base_ad'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][base_ad_str]

# b) construct each ad's initial clicks and repeats
    str = f"r_{ad_rank}"
    X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), 'impression_repeat'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)
    str = f"c_{ad_rank}"
    X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), 'previous_clicks'] = data[(data['user_visit_no'] == user_visit_no)  & (data['split'] == split_no)][str]
    return X



def calc_split_tes(data, split_no, user_visit_no, ranks_list):
    """
    This function calculates the treatment effects for the ads with ranks in "rank_list" for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
    The output is saved in columns te_1, ..., te_{max_adv_rank} of the dataframe "data"
    """
    start_time = time.perf_counter()
    for rank in ranks_list:
        X_s = construct_split_X(data, split_no, user_visit_no=user_visit_no, ad_rank=rank)
        X = construct_actual_X(data, split_no, user_visit_no=user_visit_no, ad_rank=rank)
        var_name_split = f"te_{rank}_s"
        var_name_actual = f"te_{rank}"
        if (len(data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)]) > 0):
            exec(f"data.loc[((data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)) , var_name_split] = config.cf_{rank}_s{split_no}.const_marginal_effect(X_s.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)])")
            exec(f"data.loc[((data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)) , var_name_actual] = config.cf_{rank}.const_marginal_effect(X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)])")
       # if rank % 10 == 1:
        #     print(f"rank {rank} done!")
    finish_time = time.perf_counter()
    print(f"finished calculating te's for rank {rank} in {finish_time - start_time} seconds")



def calc_base_ad_split_ctr(data, split_no, user_visit_no):
    """
    This function calculates E(y0|X=x) for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
    The output is saved in columns y_{base_ad} of the dataframe "data"
    """
    start_time = time.perf_counter()
    # # Define X variables 
    # X_s = data.loc[data['split'] == split_no, ['impression_repeat_s', 'impression_repeat_base_ad_s', 
    #           'previous_clicks_s', 'previous_clicks_base_ad_s', 'previous_clicks_all_ads_s',
    #            'total_visits_s', 
    #            'visit_s1_s', 'visit_s2_s', 'visit_s3_s', 'visit_s4_s', 'visit_s5_s',
    #            'visit_s6_s','visit_s7_s', 'visit_s8_s', 'visit_s9_s', 'visit_s10_s',
    #            'visit_s11_s','visit_s12_s', 'visit_s13_s',
    #            'sub_1_s', 'sub_2_s', 'sub_3_s', 'sub_4_s', 'sub_5_s', 
    #            'sub_6_s', 'sub_7_s', 'sub_8_s','sub_9_s', 'sub_10_s', 
    #            'sub_11_s', 'sub_12_s', 'sub_13_s',
    #            'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']]

    # # remove "_s" from column names in X to be able to run the causal forest model
    # X_s.columns = X_s.columns.str[:-2]

    # # rename var name "mobi" to 'mobile' (revert the above line)
    # X_s = X_s.rename(columns={'mobi': 'mobile', 'ho': 'hour', 'd': 'day', 'ho': 'hour', 'ads_on_pa': 'ads_on_page'})



    X = construct_X(data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)], user_visit_no=user_visit_no, ad_rank=1) # Here I use cf_1 (the first ads causal forest to estimate y0)
    tau_1 = config.cf_1.const_marginal_effect(X[data['user_visit_no'] == user_visit_no]).reshape(-1, 1)
    m_1 = config.m1.predict(X[data['user_visit_no'] == user_visit_no]).reshape(-1, 1)
    e_1 = config.e1.predict_proba(X[data['user_visit_no'] == user_visit_no]).reshape(-1, 1)
    y_0 = m_1 - tau_1 * e_1


    var_name_split = f"y_{base_ad}_s"
    var_name_actual = f"y_{base_ad}"

    if (len(data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)]) > 0):
        data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), var_name_split] = np.maximum(y_0, 0)
        data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), var_name_actual] = np.maximum(y_0, 0)
    finish_time = time.perf_counter()
    # print(f"finished calculating y0 in {finish_time - start_time} seconds")



def calc_split_ctrs(data, vals_data, split_no, user_visit_no, ranks_list):
    """
    This function calculates the click rates of all ads for the subset of DataFrame "data" for which the "user_visit_no" is a specific number by adding y_{base_ad} and treatment effects.
    The output is saved in columns y_1, ..., y_{max_adv_rank} of the dataframe "data"
    """
    start_time = time.perf_counter()
    for rank in ranks_list:
        y_var_name_split = f'y_{rank}_s'
        te_var_name_split = f'te_{rank}_s'
        y_base_ad_split = f'y_{base_ad}_s'
        rev_var_name_split = f'rev_{rank}_s'
        y_var_name_actual = f'y_{rank}'
        te_var_name_actual = f'te_{rank}'
        y_base_ad_actual = f'y_{base_ad}'
        rev_var_name_actual = f'rev_{rank}'
        


        if (len(data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)]) > 0):
            # calc ctr for split data
            data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_var_name_split] = data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), te_var_name_split] + data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_base_ad_split]
            # set y_{rank} to 0 if it is negative
            data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_var_name_split] = data.loc[data['user_visit_no'] == user_visit_no, y_var_name_split].apply(lambda x: max(x, 0))
            # calc ctr for actual data
            data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_var_name_actual] = data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), te_var_name_actual] + data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_base_ad_actual]
            # set y_{rank} to 0 if it is negative
            data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_var_name_actual] = data.loc[data['user_visit_no'] == user_visit_no, y_var_name_actual].apply(lambda x: max(x, 0))

            # revenues:
            
            data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), rev_var_name_split] = data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_var_name_split] * vals_data.loc[vals_data['advertiser_rank'] == rank].advertiser_val_cents.squeeze()     # squeeze() converts pandas Series to scalar
            data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), rev_var_name_actual] = data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), y_var_name_actual] * vals_data.loc[vals_data['advertiser_rank'] == rank].advertiser_val_cents.squeeze()


    finish_time = time.perf_counter()
    # print(f"finished calculating y_i's in {finish_time - start_time} seconds")





def create_chosen_split_ad_vars(data):
    """
    This functions initializes three sets of variable in the dataframe "data":
    1) chosen_ad_{ad}: shows the rank of the the top {ad} chosen ad, ex: chosen_ad_1 is the rank of the top ad chosen to be shown
    2)chosen_ad_y_{ad}: shows the corresponding treatment effect of that ad
    Initially, all these columns are NaN
    3) num_ads:  number of ads to be shown (currently nan)

    Inputs:
    - data: the dataframe

    """
    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_{ad}"
        data.loc[:, var_name] = np.nan


    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_y_{ad}_s" 
        data.loc[:, var_name] = np.nan


    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_y_{ad}" 
        data.loc[:, var_name] = np.nan

    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_te_{ad}" 
        data.loc[:, var_name] = np.nan

    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_rev_{ad}" 
        data.loc[:, var_name] = np.nan

    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_rev_{ad}_s" 
        data.loc[:, var_name] = np.nan

    for ad in range(1, config.max_ads_per_page + 1):
        var_name = f"chosen_ad_click_dummy_{ad}"
        data.loc[:, var_name] = np.nan
    data.loc[:, 'num_ads'] = np.nan






def find_optimal_split_ads(row, criteria):
    """
    This functions calculates optimal ads (based on highest treatment effects) to be shown to the impression in each row. based on the calculated treatment effects y_i s
    Inputs: 
        - row: the row of the dataframe that it is applied to
        it has to include indices y_cols and "ads_on_page" (determines how many ads to choose)
    
    Returns: 
        - chosen_ads: a list of ads to be shown
        - chosen_ad_ys: a list of the corresponding treatment effects
    """
    chosen_ad_ys_actual = []
    chosen_ad_revs_actual = []
    y_cols = row.filter(regex=r'^y_.*_s$', axis=0)
    rev_cols = row.filter(regex=r'^rev_.*_s', axis=0)

    # sort the values by the value of the criteria
    if criteria == "CTR":
        sorted_ads = row[y_cols.index].sort_values(ascending=False).index.to_list()
        l = min(row['ads_on_page'], config.max_ads_per_page)    # number of ads to be shown on each visit
        chosen_ads = sorted_ads[0 : l]
        chosen_ads = [int(element[2: -2]) for element in chosen_ads] # this will turn y_25_s into 25!

    if criteria == "revenue":
        sorted_ads = row[rev_cols.index].sort_values(ascending=False).index.to_list()
        l = min(row['ads_on_page'], config.max_ads_per_page)    # number of ads to be shown on each visit  
        chosen_ads = sorted_ads[0 : l]
        chosen_ads = [int(element[4: -2]) for element in chosen_ads] # this will turn rev_25_s into 25!


    # creates a list of chosen ad ranks
    chosen_ad_ys_split = y_cols.sort_values(ascending=False).values[0:l]
    chosen_ad_revs_split = rev_cols.sort_values(ascending=False).values[0:l]

    for chosen_ad in chosen_ads:
        y_var_name = f"y_{chosen_ad}"
        chosen_ad_ys_actual.append(row[y_var_name])

        rev_var_name = f"rev_{chosen_ad}"
        chosen_ad_revs_actual.append(row[rev_var_name])

    return chosen_ads, chosen_ad_ys_split, chosen_ad_revs_split, chosen_ad_ys_actual, chosen_ad_revs_actual



    
def create_chosen_ad_columns_split(data, split_no, user_visit_no, criteria): 
    """
    This function finds the optimal ads for the subsection of "data" for which user_visit_no == user_visit_no
    The chosen ads and their corresponding click rates are saved in 'chosen_ad_{ad}' and 'chosen_ad_y_{ad}'
    """
    # select treatment effect columns
    # te_cols = data.loc[0: 1, :].filter(regex="^te_", axis=1).columns
    # select ctr columns:
    # y_cols = data.loc[0: 1, :].filter(regex="^y_", axis=1).columns
    y_cols = data.columns[data.columns.str.match(r"^y_.*_s$")]

    for index, row in data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)].iterrows():
        
        chosen_ads, chosen_ad_ys_split, chosen_ad_revs_split, chosen_ad_ys_actual, chosen_ad_revs_actual = find_optimal_split_ads(row, criteria)
        chosen_ads = [int(element) for element in chosen_ads]
        l = len(chosen_ads)
        last_chosen_ad_name = f"chosen_ad_{l}"
        # last_chosen_ad_te_name = f"chosen_ad_te_{l}"
        last_chosen_ad_y_name_split = f"chosen_ad_y_{l}_s"
        last_chosen_ad_y_name_actual = f"chosen_ad_y_{l}"
        last_chosen_ad_rev_name_split = f"chosen_ad_rev_{l}_s"
        last_chosen_ad_rev_name_actual = f"chosen_ad_rev_{l}"
        data.loc[index, 'chosen_ad_1': last_chosen_ad_name] = chosen_ads
        data.loc[index, 'chosen_ad_y_1_s' : last_chosen_ad_y_name_split] = chosen_ad_ys_split
        data.loc[index, 'chosen_ad_rev_1_s' : last_chosen_ad_rev_name_split] = chosen_ad_revs_split
        data.loc[index, 'chosen_ad_y_1' : last_chosen_ad_y_name_actual] = chosen_ad_ys_actual
        data.loc[index, 'chosen_ad_rev_1' : last_chosen_ad_rev_name_actual] = chosen_ad_revs_actual
        data.at[index, 'num_ads'] = int(l)
        # if index % 10000 == 0:
        #     print(f"index {index} done!")






       

# def calc_base_ad_actual_ctr(data, split_no, user_visit_no):
#     """
#     This function calculates E(y0|X=x) for the subset of DataFrame "data" for which the "user_visit_no" is a specific number.
#     The output is saved in columns y_{base_ad} of the dataframe "data"
#     """
#     start_time = time.perf_counter()
#     # Define X variables (Note that I am not using previous_clicks and i mpression_repeat variables here, because I'm only using base ad repeats and clicks here)
#     X = data[['previous_clicks_all_ads',
#         'impression_repeat_base_ad', 'previous_clicks_base_ad', 'total_visits',
#         'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5', 'visit_s6',
#         'visit_s7', 'visit_s8', 'visit_s9', 'visit_s10', 'visit_s11',
#         'visit_s12', 'visit_s13', 'visit_s14', 'visit_s15', 'visit_s16',
#         'visit_s17', 'visit_s18', 'visit_s19', 'visit_s20', 'visit_s21',
#         'visit_s22', 'visit_s23', 'visit_s24', 'visit_s25', 'visit_s26',
#         'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8',
#         'sub_9', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15',
#         'sub_16', 'sub_17', 'sub_18', 'sub_19', 'sub_20', 'sub_21', 'sub_22',
#         'sub_23', 'sub_24', 'sub_25', 'sub_26', 'mobile']]
    

#     var_name = f"y_{base_ad}"
#     if (len(data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)]) > 0):
#         data.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no), var_name] = config.base_ad_y_model.predict(X.loc[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)])
#     finish_time = time.perf_counter()
#     # print(f"finished calculating y0 in {finish_time - start_time} seconds")



# def calc_actual_tes_for_chosen_ads(data, index):
#     tes_list =[]
#     for chosen_ad_no in range(1, int(data.loc[index, 'num_ads']) + 1):
#         # var_name = f"chosen_ad_te_{chosen_ad_no}"
#         chosen_ad_var = f"chosen_ad_{chosen_ad_no}"
#         chosen_ad = int(data.at[index, chosen_ad_var])
#         X = data.loc[index: index, ['impression_repeat', 'previous_clicks', 'previous_clicks_all_ads',
#         'impression_repeat_base_ad', 'previous_clicks_base_ad', 'total_visits',
#         'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5', 'visit_s6',
#         'visit_s7', 'visit_s8', 'visit_s9', 'visit_s10', 'visit_s11',
#         'visit_s12', 'visit_s13', 'visit_s14', 'visit_s15', 'visit_s16',
#         'visit_s17', 'visit_s18', 'visit_s19', 'visit_s20', 'visit_s21',
#         'visit_s22', 'visit_s23', 'visit_s24', 'visit_s25', 'visit_s26',
#         'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8',
#         'sub_9', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15',
#         'sub_16', 'sub_17', 'sub_18', 'sub_19', 'sub_20', 'sub_21', 'sub_22',
#         'sub_23', 'sub_24', 'sub_25', 'sub_26', 'mobile']]
        
#         # #################################### this is for fixing the missing variable sub_24, sub_25, sub_26 on the second split when training the model. Remove this when you fix this problem:
#         # if row['split'] == 2:
#         #     X = X.drop(['sub_24', 'sub_25', 'sub_26'])

#     # a) construct base ad's initial clicks and repeats
#         base_ad_str = f"r_{base_ad}"
#         X['impression_repeat_base_ad'] = data.loc[index, base_ad_str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)

#         base_ad_str = f"c_{base_ad}"
#         X['previous_clicks_base_ad'] =data.loc[index, base_ad_str]

#     # b) construct chosen ad's initial clicks and repeats
#         str = f"r_{chosen_ad}"
#         X['impression_repeat'] = data.loc[index, str] + 1  # +1 is because r_* shows previous impressions, but impression repeat is the number of repeats (including current one)
#         str = f"c_{chosen_ad}"
#         X['previous_clicks'] = data.loc[index, str]
#         if chosen_ad != base_ad:
#             exec(f"tes_list.append(config.cf_{chosen_ad}.const_marginal_effect(X))")
#         else:
#             tes_list.append(np.array([[0]]))
#     return np.concatenate(tes_list).flatten()



# def calc_actual_ctrs_for_chosen_ads(data, split_no, user_visit_no):
#     print("updated")
#     for index, row in (data[(data['split'] == split_no) & (data['user_visit_no'] == user_visit_no)]).iterrows():
#         # if (index % 100 == 0):
#         #     print(index)
#         tes_list =  calc_actual_tes_for_chosen_ads(data, index)
#         l = len(tes_list)
#         last_chosen_ad_te_name = f"chosen_ad_te_{l}"
#         data.loc[index, 'chosen_ad_te_1' : last_chosen_ad_te_name] = tes_list
#         last_chosen_ad_y_name = f"chosen_ad_y_{l}"
#         base_ad_ctr_var = f"y_{base_ad}"
#         data.loc[index, 'chosen_ad_y_1' : last_chosen_ad_y_name] = data.loc[index, 'chosen_ad_te_1' : last_chosen_ad_te_name] + data.loc[index, base_ad_ctr_var]
        



def update_repeats_on_main_and_split(data, split_no, user_visit_no):

    """
    This function updates the number of previous impression on data after user visit number user_visit_no.
    For example, after a user visits a page for the first time, and observes optimal ads (say ads 2, 5, 10), the initial impressions for all subsequent visits of that user, the number of previous impressions on ads 2, 5, 10 increases by 1. 
    """
    for index, row in data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)].iterrows():

        for chosen_ad_no in range(1, int(row['num_ads']) + 1):
            var_name = f"chosen_ad_{chosen_ad_no}"
            chosen_ad = int(row[var_name])
            col_name_main = f'r_{chosen_ad}'
            col_name_split = f'r_{chosen_ad}_s'
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name_main] = row[col_name_main] + 1 # update actual repeats on all subsequent impressions of the user
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no']) & (data['split'] == row['split'])), col_name_split] = row[col_name_split] + 1 # update split repeats on subsequent impressions of the user only if it is on the same split (platform)



def update_clicks_on_main_and_split(data, split_no, user_visit_no):
    """
    This function updates the number of previous clicks on data after user visit number user_visit_no.
    For example, after a user visits a page for the first time, and clicks on ad 5, c_5 increases by 1 for all subsequent user impressions. 
    It also updates the column "previous_clicks_all_ads"
    """

    for index, row in data[(data['user_visit_no'] == user_visit_no) & (data['split'] == split_no)].iterrows():
        total_clicks_on_impression = 0
        for chosen_ad_no in range(1, int(row['num_ads']) + 1):
            var_name = f"chosen_ad_{chosen_ad_no}"
            chosen_ad = int(row[var_name])
            ctr_var = f'chosen_ad_y_{chosen_ad_no}'
            col_name_main = f'c_{chosen_ad}' # the column name to be updated (if ad 5 is clicked on, c_5 will increase by 1 for all subsequent impressions)
            col_name_split = f'c_{chosen_ad}_s' # the column name to be updated (if ad 5 is clicked on, c_5_s will increase by 1 for all subsequent impressions)
            click_dummy_var =f'chosen_ad_click_dummy_{chosen_ad_no}'
            rand_click = np.random.rand()   # a random number simulating user's click. User will click if rand_click < y_{chosen_ad}
            # print(data.at[index, ctr_var])
            data.loc[index, click_dummy_var] = int(rand_click <= row[ctr_var])
            total_clicks_on_impression += data.loc[index, click_dummy_var]
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name_main] = int(row[col_name_main] + data.loc[index, click_dummy_var])
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no']) & (data['split'] == row['split'])), col_name_split] = int(row[col_name_split] + data.loc[index, click_dummy_var]) # update only if it is on the same split (platform)
        data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), 'previous_clicks_all_ads'] = int(row['previous_clicks_all_ads'] + total_clicks_on_impression)
        data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no']) & (data['split'] == row['split'])), 'previous_clicks_all_ads_s'] = int(row['previous_clicks_all_ads'] + total_clicks_on_impression)  # update only if it is on the same split (platform)





# the only difference between the sqrt(n) and the other case is that clicks and repeats are updated on all data sets (main, and both splits) in the sqrt(n) case
# but in the normal case the competing platform is not observing the user activity on the other platform.
def update_repeats_on_main_and_split_sqrt_n(data, user_visit_no):

    """
    This function updates the number of previous impression on data after user visit number user_visit_no.
    For example, after a user visits a page for the first time, and observes optimal ads (say ads 2, 5, 10), the initial impressions for all subsequent visits of that user, the number of previous impressions on ads 2, 5, 10 increases by 1. 
    """
    for index, row in data[(data['user_visit_no'] == user_visit_no)].iterrows():

        for chosen_ad_no in range(1, int(row['num_ads']) + 1):
            var_name = f"chosen_ad_{chosen_ad_no}"
            chosen_ad = int(row[var_name])
            col_name_main = f'r_{chosen_ad}'
            col_name_split = f'r_{chosen_ad}_s'
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name_main] = row[col_name_main] + 1 # update actual repeats on all subsequent impressions of the user
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no']) ), col_name_split] = row[col_name_split] + 1 # update split repeats on subsequent impressions of the user under split



def update_clicks_on_main_and_split_sqrt_n(data, user_visit_no):
    """
    This function updates the number of previous clicks on data after user visit number user_visit_no.
    For example, after a user visits a page for the first time, and clicks on ad 5, c_5 increases by 1 for all subsequent user impressions. 
    It also updates the column "previous_clicks_all_ads"
    """

    for index, row in data[(data['user_visit_no'] == user_visit_no)].iterrows():
        total_clicks_on_impression = 0
        for chosen_ad_no in range(1, int(row['num_ads']) + 1):
            var_name = f"chosen_ad_{chosen_ad_no}"
            chosen_ad = int(row[var_name])
            ctr_var = f'chosen_ad_y_{chosen_ad_no}'
            col_name_main = f'c_{chosen_ad}' # the column name to be updated (if ad 5 is clicked on, c_5 will increase by 1 for all subsequent impressions)
            col_name_split = f'c_{chosen_ad}_s' # the column name to be updated (if ad 5 is clicked on, c_5_s will increase by 1 for all subsequent impressions)
            click_dummy_var =f'chosen_ad_click_dummy_{chosen_ad_no}'
            rand_click = np.random.rand()   # a random number simulating user's click. User will click if rand_click < y_{chosen_ad}
            # print(data.at[index, ctr_var])
            data.loc[index, click_dummy_var] = int(rand_click <= row[ctr_var])
            total_clicks_on_impression += data.loc[index, click_dummy_var]
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name_main] = int(row[col_name_main] + data.loc[index, click_dummy_var])
            data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), col_name_split] = int(row[col_name_split] + data.loc[index, click_dummy_var]) 
        data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), 'previous_clicks_all_ads'] = int(row['previous_clicks_all_ads'] + total_clicks_on_impression)
        data.loc[((data['global_token_new'] == row['global_token_new']) & (data['user_visit_no'] > row['user_visit_no'])), 'previous_clicks_all_ads_s'] = int(row['previous_clicks_all_ads'] + total_clicks_on_impression) 



