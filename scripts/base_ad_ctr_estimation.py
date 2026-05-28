# ## It took 35 mins to run this.

from adsim import config
from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.utils import *
start_time_1 = time.perf_counter()
# read data
file_name = "Estimation Data - Full Model - Monopoly.dta"
data = pd.read_stata(DATA_DIR / "Full Model" / file_name)


# load cf_1
cf_1 = joblib.load(RESULTS_DIR / "Full Model" / "Monopoly" / "CF - Rank 1.pkl")



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
_full_model_dir = RESULTS_DIR / "Full Model"
_full_model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(m1, _full_model_dir / "m1.pkl")
joblib.dump(e1, _full_model_dir / "e1.pkl")

finish_time_1 = time.perf_counter()
print(f"finished in {finish_time_1 - start_time_1} seconds")