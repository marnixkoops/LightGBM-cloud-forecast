from datetime import datetime

# General settings
RUNTAG = 'LGBM_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
# RUNTAG = 'ALL_PRODUCTS_ALL_LAGS'
# RUNTAG = 'PROMO_SUBSET_ALL_LAGS'
# RUNTAG = 'OOS_SUBSET_ALL_LAGS'
# RUNTAG = 'NORMAL_SUBSET_ALL_LAGS'

COMPUTE_SHAP = False

# GCS settings
PROJECT = 'coolblue-bi-platform-dev'
BUCKET = 'coolblue-ds-demand-forecast-dev'
DATA_DIR = 'madeleine/lightgbm/data'
FEATURES_DIR = 'madeleine/lightgbm/features_per_fold'
MODEL_DIR = 'madeleine/lightgbm/model'
RESULTS_DIR = 'madeleine/lightgbm/results'
PLOTS_DIR = 'madeleine/lightgbm/plots'
CODE_DIR = 'madeleine/lightgbm/code'

# Feature-construction settings
LAGS = [7, 8, 9, 14, 21]
NUM_FOLDS = 10

# Model settings
# Categorical features
CAT_FEATURES = ['product_id', 'product_type_id', 'brand_id', 'manufacturer_id', 'product_group_id', 'team_id',
                'subproduct_type_id', 'month', 'weekday', 'dayofmonth', 'weekofyear', 'year', 'dayofyear']

# Parameters quite randomly chosen for now, should be optimized at some point
PARAMS = {
    'nthread': -1,
    'boosting_type': 'gbdt',
    'objective': 'huber',  # set to 'None' if custom objective is supplied in the lgb.train call below
    'metric': 'huber',  # set to 'None' if custom metric is supplied in the lgb.train call below
    'learning_rate': 0.1,
    'max_depth': 24,
    'num_leaves': 82,
    'feature_fraction': 0.8,
    'subsample': 0.8,
    'subsample_freq': 10,
    # 'min_data_in_leaf': 25,
    # 'min_split_gain': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'verbose': -1
}
