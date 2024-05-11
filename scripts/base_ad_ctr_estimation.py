## It took 12 mins to run this.

# import config
from utils import *



start_time_1 = time.perf_counter()
# read data
file_name = "Estimation Data by Subject - Last Two Days Binary.dta"
file_dir = "..\\data\\"
file_dir_name = file_dir + file_name
data = pd.read_stata(file_dir_name)


# only keep the base ad
data = data[data['advertiser_rank'] == base_ad]

# Define X variables (Note that I am not using previous_clicks and i mpression_repeat variables here, because I'm only using base ad repeats and clicks here)
X = data[['previous_clicks_all_ads',
    'impression_repeat_base_ad', 'previous_clicks_base_ad', 'total_visits',
    'visit_s1', 'visit_s2', 'visit_s3', 'visit_s4', 'visit_s5', 'visit_s6',
    'visit_s7', 'visit_s8', 'visit_s9', 'visit_s10', 'visit_s11',
    'visit_s12', 'visit_s13', 'visit_s14', 'visit_s15', 'visit_s16',
    'visit_s17', 'visit_s18', 'visit_s19', 'visit_s20', 'visit_s21',
    'visit_s22', 'visit_s23', 'visit_s24', 'visit_s25', 'visit_s26',
    'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8',
    'sub_9', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15',
    'sub_16', 'sub_17', 'sub_18', 'sub_19', 'sub_20', 'sub_21', 'sub_22',
    'sub_23', 'sub_24', 'sub_25', 'sub_26', 'mobile']]

Y = data['is_clicked']


# Define the hyperparameters to search over
param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [1000 , 2000, 3000, 5000]
}


start_time = time.perf_counter()
y0_model = RandomForestRegressor(verbose=0, n_jobs=30)

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=y0_model, param_grid=param_grid, cv=5, verbose=1)
grid_search.fit(X, Y)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
finish_time = time.perf_counter()
print(f"finished tuning the y0 model in {finish_time - start_time} seconds")

base_ad_y_model = best_estimator

# save the model
file_name = f"..\\results\\main_scenario\\base_ad_y_model.pkl"
joblib.dump(base_ad_y_model, file_name)
finish_time_1 = time.perf_counter()
print(f"finished in {finish_time_1 - start_time_1} seconds")