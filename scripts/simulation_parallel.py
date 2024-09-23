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
import multiprocessing
import pickle

from propensity_model import PropensityModel
from utils import *
import config




# For ignoring the warnings
from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

for rank in ranks_list:
    cf =  joblib.load(f'..\\results\\Full Model\\Monopoly\\CF - Rank {rank}.pkl')
    exec(f"cf_{rank} = cf")
    if rank % 20 == 0:
        print(f"rank {rank} model loaded!")

base_ad = 50
max_adv_rank = 100
max_visit_no = 100 # max number of page visits by each user

# read data
data = pd.read_stata("..\\data\\Simulation Data - Last 2 Days - Merged Subjects Subsample.dta")

start_time_main = time.perf_counter()

# Chunk the data
chunk_users_num = 820000
n_chunks = int(data.global_token_new.max() / chunk_users_num) + 1

data['chunk'] = ((data['global_token_new'] / chunk_users_num).astype(int) + 1)

data_chunks = []

# create data chunks: data_chunk_1, ...
for chunk in range(1, n_chunks + 1):
    # var_name = f"data_chunk_{chunk}"
    exec(f"data_chunk_{chunk} = data[data['chunk']==chunk]")
    exec(f"data_chunks.append(data_chunk_{chunk})")


start_time = time.perf_counter()
def simulate(data):
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
        calc_ctrs(data, user_visit_no=i)

        # 2) determine what ads are chosen
        # a. create empty columns in the dataframe to fill later
        start_time_2 = time.perf_counter()
        # find the optimal ads and save them and their corresponding ctr's in the dataframe
        create_chosen_ad_columns(data, user_visit_no=i)
        finish_time_2 = time.perf_counter()

    
        # 3) Update repeats and clicks for the next impressions
        # start_time_1 = time.perf_counter()
        update_repeats(data, user_visit_no=i)
        update_clicks(data, user_visit_no=i)

        finish_time_1 = time.perf_counter()

        print(f"Repeat {i} finished in {finish_time_1 - start_time_1} seconds!")


    finish_time = time.perf_counter()
    print(f"All Repeats finished in {finish_time - start_time} seconds!")
    return data

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    with multiprocessing.Pool(processes=9) as pool:
        results = pool.map(simulate, data_chunks)  # Parallel execution

        main_df = pd.DataFrame()
        for result_df in results: 
            main_df = pd.concat([main_df, result_df], ignore_index=True)
        main_df.to_stata("..\\results\\Simluation Results - Subsample.dta")
# finish_time = time.perf_counter()
# print(f"Merging files finished in {finish_time - start_time} seconds!")


# finish_time_main = time.perf_counter()
# print(f"Chunk {chunk} out of {n_chunks} finished in {finish_time_main - start_time_main} seconds!")



