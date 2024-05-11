# %%
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

from utils import *



# %%
base_ad = 50
max_adv_rank = 100


# %%
with open("..\\results\main_scenario\\ranks_list.pickle", "rb") as file:
    ranks_list = pickle.load(file)

ranks_list.pop(0)
ranks_list.pop(-1)

# %%
# read data
data = pd.read_stata("..\\data\\Simulation Data - Last 2 Days.dta")



# load causal forests
for rank in ranks_list:
    cf =  joblib.load(f'..\\results\\main_scenario\\CF - Rank {rank}.pkl')
    if rank % 20 == 0:
        print(f"rank {rank} model loaded!")
    exec(f"cf_{rank} = cf")




# %%
calc_tes(data=data, user_visit_no=1, ranks_list=ranks_list)

# # %%
# # select treatment effect columns
# te_cols = data.loc[0: 1, :].filter(regex="^te_", axis=1).columns

# # %%
# max_ads_per_page = 15



# # %%
# data_mini = data.loc[0 : 10_000, :]

# # %%
# create_chosen_ad_columns(data=data_mini, user_visit_no=1)

# # %%



