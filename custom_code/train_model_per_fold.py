import subprocess
import tempfile

import gc
import lightgbm as lgb
import pandas as pd

from custom_code.upload_file_to_gcs import upload_file_to_gcs
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.settings import PARAMS, CAT_FEATURES, NUM_FOLDS, FEATURES_DIR, MODEL_DIR, RUNTAG, BUCKET, PROJECT


def train_model_per_fold():
    for fold in list(range(0, NUM_FOLDS-1)):
        print('Training model for fold {}'.format(fold))

        print('Reading features slices')
        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/train_x_complete_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        train_x = pd.read_hdf(file_location, 'train_x')
        subprocess.call(['rm', '-f', file_location])

        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/test_x_complete_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        test_x = pd.read_hdf(file_location, 'test_x')
        subprocess.call(['rm', '-f', file_location])

        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/train_y_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        train_y = pd.read_hdf(file_location, 'train_y')
        subprocess.call(['rm', '-f', file_location])

        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/test_y_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        test_y = pd.read_hdf(file_location, 'test_y')
        subprocess.call(['rm', '-f', file_location])

        # Specify numeric and categorical features
        features_names = [f for f in train_x.columns if f not in ['date', 'actual', 'on_stock']]

        # Create lgb dataframes and train model
        print('Building lgb datsets')
        lgb_train = lgb.Dataset(train_x[features_names], categorical_feature=CAT_FEATURES, label=train_y, free_raw_data=False)
        lgb_test = lgb.Dataset(test_x[features_names], categorical_feature=CAT_FEATURES, label=test_y, free_raw_data=False)
        del train_x, test_x, train_y, test_y

        print('Training model')
        booster = lgb.train(
            PARAMS,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_test],
            categorical_feature=CAT_FEATURES,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        # Save booster object to disk
        print('Writing model to GCS')
        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            booster.save_model('{}.txt'.format(tf.name))
            upload_file_to_gcs(PROJECT, BUCKET, '{}.txt'.format(tf.name), '{}/booster_{}_{}.txt'.format(MODEL_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.txt'.format(tf.name)])
        del booster, lgb_train, lgb_test
        gc.collect()
