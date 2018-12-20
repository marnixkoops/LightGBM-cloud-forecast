import tempfile
from datetime import datetime
import subprocess

import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import gc

from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.upload_file_to_gcs import upload_file_to_gcs
from custom_code.settings import CAT_FEATURES, NUM_FOLDS, RUNTAG, PROJECT, BUCKET, MODEL_DIR, FEATURES_DIR, RESULTS_DIR, COMPUTE_SHAP


def predict_and_save_results_per_fold():
    for fold in list(range(0, NUM_FOLDS-1)):
        print('Predicting results for fold {}'.format(fold))

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

        print('Loading lgb trained model')
        file_location = '{}/booster_{}_{}.txt'.format(MODEL_DIR, fold, RUNTAG)
        file_location = download_file_from_gcs(PROJECT, BUCKET, file_location)
        booster = lgb.Booster(model_file=file_location)
        subprocess.call(['rm', '-f', file_location])

        print('Predicting train data')
        train_preds = booster.predict(lgb_train.data, num_iteration=booster.best_iteration)

        print('Predicting test data')
        test_preds = booster.predict(lgb_test.data, num_iteration=booster.best_iteration)

        print('Writing results for fold {} at {}'.format(fold, datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        fold_result_df = pd.DataFrame()
        fold_result_df['product_id'] = train_x['product_id'].append(test_x['product_id'])
        fold_result_df['date'] = train_x['date'].append(test_x['date'])
        fold_result_df['on_stock'] = train_x['on_stock'].append(test_x['on_stock'])
        fold_result_df['fold'] = np.repeat(fold, len(train_x.index) + len(test_x.index))
        fold_result_df['actual'] = train_x['actual'].append(test_x['actual'])
        fold_result_df['lgbm'] = np.concatenate([train_preds, test_preds])
        fold_result_df['is_test'] = np.concatenate([np.repeat(False, len(train_x.index)), np.repeat(True, len(test_x.index))])
        fold_result_df = fold_result_df.sort_values(by=['product_id', 'date'])

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            fold_result_df.to_hdf('{}.h5'.format(tf.name), 'fold_result_df', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/results_{}_{}.h5'.format(RESULTS_DIR, fold, RUNTAG))
            fold_result_df.to_csv('{}.csv'.format(tf.name), index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name), '{}/results_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])

        # Compute feature importances
        gain = booster.feature_importance(importance_type='gain')
        split = booster.feature_importance(importance_type='split')
        fold_importance_df = pd.DataFrame()
        fold_importance_df['fold'] = np.repeat(fold, len(features_names))
        fold_importance_df['feature'] = features_names
        fold_importance_df['gain'] = gain
        fold_importance_df['split'] = split

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            fold_importance_df.to_hdf('{}.h5'.format(tf.name), 'fold_importance_df', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/importance_{}_{}.h5'.format(RESULTS_DIR, fold, RUNTAG))
            fold_importance_df.to_csv('{}.csv'.format(tf.name), index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name), '{}/importance_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])

        # Only compute Shapley values if specified
        if COMPUTE_SHAP:
            print('Started computing fold {} SHAP values at {}'.format(fold, datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
            explainer_fold = shap.TreeExplainer(booster)
            shap_values = explainer_fold.shap_values(train_x[features_names])

            shap_df = pd.DataFrame()
            shap_df['fold'] = np.repeat(fold, len(shap_values))
            shap_df['date'] = train_x['date'].append(test_x['date'])
            for col_num, features_name in enumerate(features_names):
                shap_df[features_name] = shap_values[:, col_num]

            with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
                shap_df.to_hdf('{}.h5'.format(tf.name), 'shap_df', index=False)
                upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/shap_{}_{}.h5'.format(RESULTS_DIR, fold, RUNTAG))
                shap_df.to_csv('{}.csv'.format(tf.name), index=False)
                upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name), '{}/shap_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
            subprocess.call(['rm', '-f', tf.name])
            subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
            subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])
            del shap_df

        del train_x, test_x, train_y, test_y
        del booster, lgb_train, lgb_test, train_preds, test_preds
        del fold_result_df, fold_importance_df
        gc.collect()
