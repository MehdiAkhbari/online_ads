import joblib
import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, make_scorer, f1_score
import pandas as pd
import numpy as np


simulation = True
root_n = False
sample_size_analysis = False
subsampling_ratio = 0.8

n_jobs = 30


my_criteria = "revenue"  # "CTR" or "revenue"





max_ads_per_page = 15

split_no_1 = 7
split_no_2 = 8


# create ranks_list
with open("..\\results\main_scenario\\ranks_list.pickle", "rb") as file:
    ranks_list = pickle.load(file)

ranks_list.pop(0)
ranks_list.pop(-1)




if simulation == True:


    e1 =  joblib.load(f"..\\results\\Full Model\\e1.pkl")
    m1 =  joblib.load(f"..\\results\\Full Model\\m1.pkl")

    # import forests:
    for rank in ranks_list:
        cf =  joblib.load(f'..\\results\\Full Model\\Monopoly\\CF - Rank {rank}.pkl')
        exec(f"cf_{rank} = cf")
        if rank % 20 == 0:
            print(f"rank {rank} model loaded!")

    # if root_n == False:
    #     # import forests:
    #     for rank in ranks_list:
    #         cf =  joblib.load(f'..\\results\\Full Model\\Split {split_no_1}\\CF - Rank {rank}.pkl')
    #         exec(f"cf_{rank}_s{split_no_1} = cf")
    #         if rank % 20 == 0:
    #             print(f"rank {rank} model loaded!")


    #     for rank in ranks_list:
    #         cf =  joblib.load(f'..\\results\\Full Model\\Split {split_no_2}\\CF - Rank {rank}.pkl')
    #         exec(f"cf_{rank}_s{split_no_2} = cf")
    #         if rank % 20 == 0:
    #             print(f"rank {rank} model loaded!")




    if root_n == True:
        # import forests:
        for rank in ranks_list:
            cf =  joblib.load(f'..\\results\\Full Model\\Split {split_no_1} - Root N\\CF - Rank {rank}.pkl')
            exec(f"cf_{rank}_s{split_no_1} = cf")
            if rank % 20 == 0:
                print(f"rank {rank} model loaded!")


        for rank in ranks_list:
            cf =  joblib.load(f'..\\results\\Full Model\\Split {split_no_2} - Root N\\CF - Rank {rank}.pkl')
            exec(f"cf_{rank}_s{split_no_2} = cf")
            if rank % 20 == 0:
                print(f"rank {rank} model loaded!")


    if sample_size_analysis == True:
        dir = "..\\results\\Full Model\\Root N - Random"
        full_dir = dir + f"\\Subsampling Ratio = {subsampling_ratio}\\"
        for rank in ranks_list:
            file_name = f'CF - Rank {rank}.pkl'
            cf =  joblib.load(full_dir + file_name)
            exec(f"cf_{rank}_sub = cf")
            if rank % 20 == 0:
                print(f"rank {rank} model loaded!")
                
