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
# import multiprocess as mp
import pickle

import sys
# sys.setrecursionlimit(10000)

n_processes = 3








from utils import *


criteria = config.my_criteria

# For ignoring the warnings
from warnings import simplefilter, filterwarnings
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None
filterwarnings("ignore", message="Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1")




# read data
data = pd.read_stata("..\\data\\Full Model\\Simulation Data - Full Model - Monopoly - Subsample.dta")
vals_data = pd.read_stata(f"..\\data\\Full Model\\Advertiser Valuations.dta")



start_time_main = time.perf_counter()


# Chunk the data
chunk_users_num = chunk_users_num = 1620000 / n_processes
n_chunks = int(data.global_token_new.max() / chunk_users_num) + 1

data['chunk'] = ((data['global_token_new'] / chunk_users_num).astype(int) + 1)

data_chunks = []

# create data chunks: data_chunk_1, ...
for chunk in range(1, n_chunks + 1):
    # var_name = f"data_chunk_{chunk}"
    exec(f"data_chunk_{chunk} = data[data['chunk']==chunk]")
    exec(f"data_chunks.append(data_chunk_{chunk})")

print("Data Chunks Created")

start_time = time.perf_counter()


# def simulate_monopoly(data):
#     # file_name = f"data_chunk_{chunk}"
#     # create empty columns in the dataframe to fill later
#     create_chosen_ad_vars(data)
#     # print(f"\n\n\n=======> Chunk #{chunk}")
#     start_time = time.perf_counter()


#     for i in range(1, max_visit_no + 1):

#         start_time_1 = time.perf_counter()
#         print(f"\n\n --->Repeat #{i}:")
#         # 1) calculate treatment effects, and base ad ctr, then sum them sup and create ctrs for all ads
#         # start_time = time.perf_counter()
#         calc_tes(data, user_visit_no=i, ranks_list=config.ranks_list)
#         calc_base_ad_ctr(data, user_visit_no=i)
#         calc_ctrs(data, user_visit_no=i)

#         # 2) determine what ads are chosen
#         # a. create empty columns in the dataframe to fill later
#         start_time_2 = time.perf_counter()
#         # find the optimal ads and save them and their corresponding ctr's in the dataframe
#         create_chosen_ad_columns(data, user_visit_no=i)
#         finish_time_2 = time.perf_counter()

    
#         # 3) Update repeats and clicks for the next impressions
#         # start_time_1 = time.perf_counter()
#         update_repeats(data, user_visit_no=i)
#         update_clicks(data, user_visit_no=i)

#         finish_time_1 = time.perf_counter()

#         print(f"Repeat {i} finished in {finish_time_1 - start_time_1} seconds!")

#     finish_time = time.perf_counter()
#     print(f"All Repeats finished in {finish_time - start_time} seconds!")
#     return data


    # finish_time = time.perf_counter()
    # print(f"All Repeats finished in {finish_time - start_time} seconds!")
    # return data

# if __name__ == '__main__':
#     multiprocessing.freeze_support() 
#     with multiprocessing.Pool(5) as pool: #num of pools

#         results = pool.map(simulate_monopoly, data_chunks)  # Parallel execution

#         main_df = pd.DataFrame()
#         for result_df in results: 
#             main_df = pd.concat([main_df, result_df], ignore_index=True)
#         main_df.to_stata("..\\results\\Full Model\\Simulation Results\\Simluation Results - Monopoly - Subsample.dta")



def simulate_monopoly_and_save_chunk(chunk_data, chunk_id, criteria):
    chunk_data = simulate_monopoly(chunk_data, vals_data, criteria) 
    # Create a unique filename for the chunk
    if criteria == "CTR":
        filename = f"..\\results\\Full Model\\Simulation Results\\Simluation Results - Monopoly - chunk {chunk_id+1}.dta"
        
    if criteria == "revenue":
        filename = f"..\\results\\Full Model\\Simulation Results\\Simluation Results - Monopoly Revenue Max - chunk {chunk_id+1}.dta"
    # Save the processed DataFrame to DTA
    chunk_data.to_stata(filename)




if __name__ == '__main__':
    multiprocessing.freeze_support() 
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.starmap(simulate_monopoly_and_save_chunk, [(chunk, i, criteria) for i, chunk in enumerate(data_chunks)])

