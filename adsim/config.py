import joblib
import pickle
import os
from adsim.constants import PATH_ROOT

max_ads_per_page = 15

split_no_1 = 7
split_no_2 = 8

# create ranks_list
path_rank_list = os.path.join(PATH_ROOT, "results", "main_scenario", "ranks_list.pickle")
with open(path_rank_list, "rb") as file:
    ranks_list = pickle.load(file)

ranks_list.pop(0)
ranks_list.pop(-1)


### comment the following lines (all of them) for estimation:

# # import forests:
# for rank in ranks_list:
#     cf =  joblib.load(f'..\\results\\main_scenario\\CF - Rank {rank}.pkl')
#     exec(f"cf_{rank} = cf")
#     if rank % 20 == 0:
#         print(f"rank {rank} model loaded!")


# # import forests:
# for rank in ranks_list:
#     cf =  joblib.load(f'..\\results\\split {split_no_1}\\CF - Rank {rank}.pkl')
#     exec(f"cf_{rank}_s1 = cf")
#     if rank % 20 == 0:
#         print(f"rank {rank} model loaded!")


# for rank in ranks_list:
#     cf =  joblib.load(f'..\\results\\split {split_no_2}\\CF - Rank {rank}.pkl')
#     exec(f"cf_{rank}_s2 = cf")
#     if rank % 20 == 0:
#         print(f"rank {rank} model loaded!")

# # import base ad ctr forest:
# base_ad_y_model = joblib.load(f"..\\results\\main_scenario\\base_ad_y_model.pkl")

