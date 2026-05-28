from utils import *


split_no = 6



class PropensityModel(BaseEstimator):
    def __init__(self):
        self.lr = LogisticRegression(max_iter=2000)

    def predict_proba(self, X, X_indices=slice(-7,-1)):
        return self.lr.predict_proba(X[:,X_indices])


    
    # X_indices are the ones that are used for the estimation of the propensity score
    def fit(self, X, y, X_indices=slice(-7,-1)):
        self.lr.fit(X[:,X_indices], y)
        return self


# Instantiate propensity_model from the PropensityModel class
propensity_model = PropensityModel()


# define the Custom Treatment Model class (its exactly the PropensityModel above. However, I had to include it because I ran the initial model with this one):
class CustomTreatmentModel(BaseEstimator):
    def __init__(self):
        self.lr = LogisticRegression(max_iter=2000)

    def predict_proba(self, X, X_indices=slice(-7,-1)):
        return self.lr.predict_proba(X[:,X_indices])


    
    # X_indices are the ones that are used for the estimation of the propensity score
    def fit(self, X, y, X_indices=slice(-7,-1)):
        self.lr.fit(X[:,X_indices], y)
        return self







from adsim.paths import DATA_DIR, RESULTS_DIR

# read data
# file_name = "Estimation Data by Subject - Last Two Days Binary - Merged Subjects.dta"
file_name = f"Estimation Data by Subject - Last Two Days Binary - split {split_no} - Merged Subjects.dta"
data = pd.read_stata(DATA_DIR / file_name)
# prepare data for estmation
prepare_data(data, base_ad=50, max_ad=100)
# extract advertiser ranks
ranks_list = extract_ranks(data)
# drop rank 0 (based ad) from the list, for the base ad we don't calculate treatment effect
_ranks_list_path = RESULTS_DIR / "main_scenario" / "ranks_list.pickle"
_ranks_list_path.parent.mkdir(parents=True, exist_ok=True)
with open(_ranks_list_path, "wb") as file:
    pickle.dump(ranks_list, file)
ranks_list.pop(0)
ranks_list.pop(-1)


for rank in ranks_list:
    start_time_1 = time.perf_counter()
    print(f"Estimating for Rank {rank}")
    # select the subset of data for current estimation:
    df = data[(data['advertiser_rank'] == 0) | (data['advertiser_rank'] == rank)].reset_index(drop=True).copy()
    (X, Y, T) = define_xyt(df)
    # find best parameters for the m model
    best_params = m_model_best_estimator(X, Y)
    # estimate the casual forest model
    # define the causal forest model
    cf = CausalForestDML(
                            model_y=RandomForestRegressor(**best_params),
                            model_t=propensity_model,
                            discrete_treatment='True',
                            criterion='het',
                            n_jobs=n_jobs,
                            n_estimators=100,
                            min_samples_split=1000,
                            max_depth=20,
                            max_samples=0.01,
                            random_state=42,
                            verbose=0   
        )
    
 # tune the model:
    start_time = time.perf_counter()

    tune_params = cf.tune(
                Y=Y,
                T=T,
                X=X,
                params=cf_param_grid)
    
    finish_time = time.perf_counter()

    print(f"finished tuning the model in {finish_time - start_time} seconds")

    # fit the model using tuned parameters:
    start_time = time.perf_counter()
    
    cf.fit(Y=Y, T=T, X=X, inference="blb", cache_values=True)
    
    finish_time = time.perf_counter()
    print(f"finished fitting the model in {finish_time - start_time} seconds")

    
    # save the model
    # _out = RESULTS_DIR / "main_scenario" / f"CF - Rank {rank}.pkl"
    _out = RESULTS_DIR / f"split {split_no}" / f"CF - Rank {rank}.pkl"
    _out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cf, _out)
    finish_time_1 = time.perf_counter()
    print(f"finished rank {rank} in {finish_time_1 - start_time_1} seconds")

