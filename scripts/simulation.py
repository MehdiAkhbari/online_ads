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
from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None



base_ad = 50
max_adv_rank = 100
max_visit_no = 100 # max number of page visits by each user

# read data
data = pd.read_stata("..\\data\\Simulation Data - Last 2 Days.dta")

start_time_main = time.perf_counter()

# Chunk the data
chunk_users_num = 300000
n_chunks = int(data.global_token_new.max() / chunk_users_num) + 1

data['chunk'] = ((data['global_token_new'] / chunk_users_num).astype(int) + 1)

# create data chunks: data_chunk_1, ...
for chunk in range(1, n_chunks + 1):
    # var_name = f"data_chunk_{chunk}"
    exec(f"data_chunk_{chunk} = data[data['chunk']==chunk]")



start_time = time.perf_counter()
for chunk in range(1, n_chunks + 1):
    file_name = f"data_chunk_{chunk}"
    # create empty columns in the dataframe to fill later
    exec(f"create_chosen_ad_vars({file_name})")
    print(f"\n\n\n=======> Chunk #{chunk}")
    start_time = time.perf_counter()


    for i in range(1, max_visit_no + 1):

        start_time_1 = time.perf_counter()
        print(f"\n\n --->Repeat #{i}:")
        # 1) calculate treatment effects, and base ad ctr, then sum them sup and create ctrs for all ads
        # start_time = time.perf_counter()
        exec(f"calc_tes(data={file_name}, user_visit_no={i}, ranks_list=config.ranks_list)")
        exec(f"calc_base_ad_ctr({file_name}, user_visit_no={i})")
        exec(f"calc_ctrs({file_name}, user_visit_no={i})")

        # 2) determine what ads are chosen
        # a. create empty columns in the dataframe to fill later
        start_time_2 = time.perf_counter()
        # find the optimal ads and save them and their corresponding ctr's in the dataframe
        exec(f"create_chosen_ad_columns({file_name}, user_visit_no={i})")
        finish_time_2 = time.perf_counter()

  
        print(f"Choosing optimal ads for repeat {i} of chunk {chunk} finished in {finish_time_2 - start_time_2} seconds!")

        # 3) Update repeats and clicks for the next impressions
        # start_time_1 = time.perf_counter()
        exec(f"update_repeats({file_name}, user_visit_no={i})")
        exec(f"update_clicks({file_name}, user_visit_no={i})")

        finish_time_1 = time.perf_counter()

  
        print(f"Repeat {i} of chunk {chunk} finished in {finish_time_1 - start_time_1} seconds!")


    finish_time = time.perf_counter()
    print(f"Chunk {chunk} out of {n_chunks} finished in {finish_time - start_time} seconds!")



# merge data chunks:
start_time = time.perf_counter()

combined_df = pd.DataFrame()

for i in range(1, n_chunks+1):  # Assuming DataFrames are named df1, df2, ..., df100
    df_name = f"data_chunk_{i}"
    try:
        df = globals()[df_name]  # Access DataFrame using its name (not generally recommended, see previous explanations)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except KeyError:
        print(f"DataFrame {df_name} not found.")  # Handle missing DataFrames

# Save to file
combined_df.to_stata("Simulated Data.dta")


finish_time = time.perf_counter()
print(f"Merging files finished in {finish_time - start_time} seconds!")


finish_time_main = time.perf_counter()
print(f"Chunk {chunk} out of {n_chunks} finished in {finish_time_main - start_time_main} seconds!")