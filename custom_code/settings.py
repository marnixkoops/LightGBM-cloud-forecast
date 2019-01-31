from datetime import datetime

# General settings
RUNTAG = '40K_WA7_NEWCV'
COMPUTE_SHAP = True

# GCS folder structure settings
PROJECT = 'coolblue-bi-platform-dev'
BUCKET = 'coolblue-ds-demand-forecast-dev'
DATA_DIR = 'marnix/lightgbm/data'
FEATURES_DIR = 'marnix/lightgbm/features_per_fold'
MODEL_DIR = 'marnix/lightgbm/model'
RESULTS_DIR = 'marnix/lightgbm/results'
PLOTS_DIR = 'marnix/lightgbm/plots'
CODE_DIR = 'marnix/lightgbm/code'

# Feature-construction settings
LAGS = [7, 8, 14]
NUM_FOLDS = 10

# LightGBM parameter settings
# Parameters quite randomly chosen for now, should be optimized at some point
PARAMS = {
    'nthread': 64, # Check if 32 is faster in prod (Real cores / 2) see https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'boosting_type': 'gbdt',
    'objective': 'huber', # Set to 'None' if custom objective is given in train call
    'metric': 'huber', # Set to 'None' if custom metric is given in train call
    'learning_rate': 0.15, # Train ~1000 rounds
    'max_depth': -1, # Unrestricted, increased from 24
    'min_data_in_leaf': 5, # Decreased from 20
    'num_leaves': 128, # Increased from 84
    'feature_fraction': 0.7, # Sample 70% of featutes
    'subsample': 1, # Was 0.9, use all data
    'subsample_freq': 0,
    'max_bin': 4096, # Increased from 255
    'reg_alpha': 0.0, # No reg
    'reg_lambda': 0.0,
    'verbose': -1
}
