# ## It took 35 mins to run this.

import config
from utils import *


start_time_1 = time.perf_counter()
# read data
file_name = "Estimation Data - Full Model - Monopoly.dta"
file_dir = "..\\data\\Full Model\\"
file_dir_name = file_dir + file_name
data = pd.read_stata(file_dir_name)
  

# load cf_1
cf_1 =  joblib.load(f'..\\results\\Full Model\\Monopoly\\CF - Rank 1.pkl')



# # only keep the base ad and ad 1
data = data[(data['advertiser_rank'] == base_ad) | (data['advertiser_rank'] == 1)]


(X, Y, T) = define_xyt(data)
# make T binary (only 0, 1)
T = T.apply(lambda x: 0 if x == 0 else 1)



# fit m and e functions
start_time_2 = time.perf_counter()
m1 = cf_1.model_y.fit(X, Y)
finish_time_2 = time.perf_counter()
print(f"y model fitted in {finish_time_2 - start_time_1} seconds")



start_time_2 = time.perf_counter()
e1 = cf_1.model_t.fit(X, T)
finish_time_2 = time.perf_counter()
print(f"y model fitted in {finish_time_2 - start_time_1} seconds")




e1 = cf_1.model_t



# # save the model
file_name = f"..\\results\\Full Model\\m1.pkl"
joblib.dump(m1, file_name)


file_name = f"..\\results\\Full Model\\e1.pkl"
joblib.dump(e1, file_name)

finish_time_1 = time.perf_counter()
print(f"finished in {finish_time_1 - start_time_1} seconds")