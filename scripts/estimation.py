from propensity_model import PropensityModel
from utils import *


split_no = 7

# read data
# file_name = f"Estimation Data - Full Model - Monopoly.dta"
file_name = f"Estimation Data - Full Model - Split {split_no}.dta"
file_dir = "..\\data\\Full Model\\"
file_dir_name = file_dir + file_name
data = pd.read_stata(file_dir_name)
# prepare data for estmation
prepare_data(data, base_ad=50, max_ad=100)
# extract advertiser ranks
ranks_list = extract_ranks(data)



with open("..\\results\\main_scenario\\ranks_list.pickle", "wb") as file:
    pickle.dump(ranks_list, file)

# drop rank 0 (based ad) from the list, for the base ad we don't calculate treatment effect
ranks_list.pop(0)
ranks_list.pop(-1)

(X, Y, T) = define_xyt(data.loc[0:2, :])

X_treat_indices = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 
               'sub_6', 'sub_7', 'sub_8','sub_9', 'sub_10', 
               'sub_11', 'sub_12', 'sub_13',
               'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page']

X_treat_indices_nums = [X.columns.get_loc(col) for col in X_treat_indices if col in X.columns]




for rank in ranks_list:
    start_time_1 = time.perf_counter()
    print(f"Estimating for Rank {rank}")
    # select the subset of data for current estimation:
    df = data[(data['advertiser_rank'] == 0) | (data['advertiser_rank'] == rank)].reset_index(drop=True).copy()

    (X, Y, T) = define_xyt(df)
    # make T binary (only 0, 1)
    T = T.apply(lambda x: 0 if x == 0 else 1)


    # find best parameters for the e model
    best_params_e, best_estimator_e = e_model_best_estimator(X, T, param_grid)

    # find best parameters for the m model
    best_params_m, best_estimator_m = m_model_best_estimator(X, Y, param_grid)



    # estimate the casual forest model
    # define the causal forest model
    cf = CausalForestDML(
                            model_y=RandomForestRegressor(**best_params_m),
                            model_t=PropensityModel(**best_params_e),
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
    # file_name = f"..\\results\\Full Model\\Monopoly\\CF - Rank {rank}.pkl"
    file_name = f"..\\results\\Full Model\\Split {split_no}\\CF - Rank {rank}.pkl"
    joblib.dump(cf, file_name)
    finish_time_1 = time.perf_counter()
    print(f"finished rank {rank} in {finish_time_1 - start_time_1} seconds")