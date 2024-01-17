import joblib
import pickle

max_ads_per_page = 15

# create ranks_list
with open("..\\results\main_scenario\\ranks_list.pickle", "rb") as file:
    ranks_list = pickle.load(file)

ranks_list.pop(0)
ranks_list.pop(-1)



# import forests:
for rank in ranks_list:
    cf =  joblib.load(f'..\\results\\main_scenario\\CF - Rank {rank}.pkl')
    exec(f"cf_{rank} = cf")
    if rank % 20 == 0:
        print(f"rank {rank} model loaded!")


# import base ad ctr forest:
base_ad_y_model = joblib.load(f"..\\results\\main_scenario\\base_ad_y_model.pkl")

