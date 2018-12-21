from datetime import datetime

# General settings
RUNTAG = 'LGBM_TEST'

COMPUTE_SHAP = False

# GCS settings
PROJECT = 'coolblue-bi-platform-dev'
BUCKET = 'coolblue-ds-demand-forecast-dev'
DATA_DIR = 'marnix/lightgbm/data'
FEATURES_DIR = 'marnix/lightgbm/features_per_fold'
MODEL_DIR = 'marnix/lightgbm/model'
RESULTS_DIR = 'marnix/lightgbm/results'
PLOTS_DIR = 'marnix/lightgbm/plots'
CODE_DIR = 'marnix/lightgbm/code'

# Feature-construction settings
LAGS = [7, 8, 9, 14, 21]
NUM_FOLDS = 5

# Model settings

# Parameters quite randomly chosen for now, should be optimized at some point
PARAMS = {
    'nthread': -1,
    'boosting_type': 'gbdt',
    'objective': 'huber',  # set to 'None' if custom objective is supplied in the lgb.train call below
    'metric': 'huber',  # set to 'None' if custom metric is supplied in the lgb.train call below
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 32,
    'feature_fraction': 1,
    'subsample': 1,
    'subsample_freq': 0,
    # 'min_data_in_leaf': 25,
    # 'min_split_gain': 0.1,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'verbose': -1
}
