from utils import *



# read data
data = pd.read_stata("..\\data\\Estimation Data by Subject - Last Two Days Binary.dta")
# prepare data for estmation
prepare_data(data, base_ad=50, max_ad=100)
# extract advertiser ranks
ranks_list = extract_ranks(data)

for rank in [99, 100]:
    start_time_1 = time.perf_counter()
    print(f"Estimating for Rank {rank}")
    # select the subset of data for current estimation:
    df = data[(data['advertiser_rank'] == 0) | (data['advertiser_rank'] == rank)].reset_index(drop=True).copy()
    (X, Y, T) = define_xyt(df)
    # estimate the casual forest model
    estimated_forest = causal_forest_estimate(X, Y, T, cf_param_grid)
    # save the model
    file_name = f"..\\results\\CF - Rank {rank}.pkl"
    joblib.dump(estimated_forest, file_name)






