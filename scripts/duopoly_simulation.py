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


import config
from utils import *



# For ignoring the warnings
from warnings import simplefilter, filterwarnings
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None



base_ad = 50
max_adv_rank = 100
max_visit_no = 100 # max number of page visits by each user

# read data
data = pd.read_stata("..\\data\\Simulation Data - Last 2 Days - Aggregate.dta")


start_time_main = time.perf_counter()




# Chunk the data
chunk_users_num = 135000
n_chunks = int(data.global_token_new.max() / chunk_users_num) + 1
data['chunk'] = ((data['global_token_new'] / chunk_users_num).astype(int) + 1)
data_chunks = []

for chunk in range(1, n_chunks + 1):
        var_name = f"data_chunk_{chunk}"
        globals()[var_name] = data[data['chunk'] == chunk]
        exec(f"data_chunks.append(data_chunk_{chunk})")

start_time = time.perf_counter()

def simulate_duopoly(data):
    # file_name = f"data_chunk_{chunk}"
    # create empty columns in the dataframe to fill later
    create_chosen_split_ad_vars(data)

    
    # print(f"\n\n\n=======> Chunk #{chunk}")
    start_time_2 = time.perf_counter()


    for i in range(1, max_visit_no + 1):

        start_time_1 = time.perf_counter()
        print(f"\n\n --->Repeat #{i}:")
        # 1) calculate treatment effects, and base ad ctr, then sum them sup and create ctrs for all ads
        start_time = time.perf_counter()
        # a) calc TEs and CTRs on s1
        calc_split_tes(data, split_no=1, user_visit_no=i, ranks_list=config.ranks_list)
        calc_base_ad_split_ctr(data,split_no=1, user_visit_no=i)
        calc_split_ctrs(data, split_no=1, user_visit_no=i, ranks_list=config.ranks_list)

        # b) calc TEs and CTRs on s2
        calc_split_tes(data, split_no=2, user_visit_no=i, ranks_list=config.ranks_list)
        calc_base_ad_split_ctr(data,split_no=2, user_visit_no=i)
        calc_split_ctrs(data, split_no=2, user_visit_no=i, ranks_list=config.ranks_list)

        finish_time = time.perf_counter()
        print(f"Step 1 of repeat {i} finished in {finish_time - start_time} seconds!")
        # 2) determine what ads are chosen
        start_time = time.perf_counter()
        # find the optimal ads and save them and their corresponding ctr's in the dataframe
        # on s1
        create_chosen_ad_columns_split(data, split_no=1, user_visit_no=i)
        

        # on s2
        create_chosen_ad_columns_split(data, split_no=2, user_visit_no=i)
        finish_time = time.perf_counter()
        print(f"Step 2 of repeat {i} finished in {finish_time - start_time} seconds!")
        # 3) Calculate actual tes and ctrs for the chosen ads
        # start_time = time.perf_counter()

        # calc_base_ad_actual_ctr(data, split_no=1, user_visit_no=i)
        # calc_base_ad_actual_ctr(data, split_no=2, user_visit_no=i)
        # finish_1 = time.perf_counter()
        # print(f"{finish_1 - start_time} seconds ")
        # calc_actual_ctrs_for_chosen_ads(data, split_no=1, user_visit_no=i)
        # calc_actual_ctrs_for_chosen_ads(data, split_no=2, user_visit_no=i)
        # finish_2 = time.perf_counter()
        # print(f"{finish_2 - finish_1} seconds ")
        # finish_time = time.perf_counter()
        # print(f"Step 3 of repeat {i} finished in {finish_time - start_time} seconds!")
        # 4) Update repeats
        start_time = time.perf_counter()

        update_repeats_on_main_and_split_sqrt_n(data, user_visit_no=i)

        finish_time = time.perf_counter()
        print(f"Step 4 of repeat {i} finished in {finish_time - start_time} seconds!")

        # 5) Update clicks
        start_time = time.perf_counter()

        update_clicks_on_main_and_split_sqrt_n(data, user_visit_no=i)     

        finish_time = time.perf_counter()
        print(f"Step 5 of repeat {i} finished in {finish_time - start_time} seconds!")
        finish_time_1 = time.perf_counter()
        print(f"Repeat {i}  finished in  {finish_time_1 - start_time_1} seconds!")

    finish_time_2 = time.perf_counter()
    data.to_stata(("..\\results\\Duopoly Simluation Results - .dta"))
    print(f"All Repeats finished in {finish_time_2 - start_time_2} seconds!")
    return data


def simulate_and_save_chunk(chunk_data, chunk_id):
    
    chunk_data = simulate_duopoly(chunk_data) 
    # Create a unique filename for the chunk
    filename = f"..\\results\\Duopoly Simluation Results - chunk {chunk_id+1}.dta"
    # Save the processed DataFrame to CSV
    chunk_data.to_stata(filename)



if __name__ == '__main__':
    multiprocessing.freeze_support() 
    # with multiprocessing.Pool(processes=5) as pool:
    #     results = pool.map(simulate_duopoly, data_chunks)  # Parallel execution
    # main_df = pd.DataFrame()
    # for result_df in results: 
    #     main_df = pd.concat([main_df, result_df], ignore_index=True)
    # main_df.to_stata("..\\results\\Duopoly Simluation Results.dta")
    with multiprocessing.Pool(processes=6) as pool:
        pool.starmap(simulate_and_save_chunk, [(chunk, i) for i, chunk in enumerate(data_chunks)])


# finish_time = time.perf_counter()
# print(f"Merging files finished in {finish_time - start_time} seconds!")

