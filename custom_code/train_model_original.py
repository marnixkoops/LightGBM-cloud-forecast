from datetime import datetime
import gc
import subprocess
import tempfile

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

from custom_code.process_features import add_fold_aware_features_faster
from custom_code.upload_file_to_gcs import upload_file_to_gcs
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code import timefold
from custom_code.settings import PROJECT, BUCKET, DATA_DIR, MODEL_DIR, RESULTS_DIR, RUNTAG, COMPUTE_SHAP, PARAMS


def train_model(features_df):
    print('Setting up LightGBM ...')
    target = features_df['actual'].copy()

    if not COMPUTE_SHAP:
        print('Warning: Shapley value computation is disabled')

    # Cross-validation setup using timefold
    min_train_size = int(0.75 * features_df.shape[0])  # 2+ years training
    min_test_size = int(0.03 * features_df.shape[0])  # 2+ months of testing
    step_size = int(0.03 * features_df.shape[0])  # 2+ month step size between folds
    timefolds = timefold.timefold(method='step', min_train_size=min_train_size, min_test_size=min_test_size, step_size=step_size)

    for fold, (train_idx, test_idx) in enumerate(timefolds.split(features_df)):
        print('Loading full dataframe inner loop ...')
        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/features_{}.h5'.format(DATA_DIR, RUNTAG))
        features_df = pd.read_hdf(file_location, 'features_df')
        subprocess.call(['rm', '-f', file_location])
        print('Train idx: {} to {}, test idx: {} to {}'.format(train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

        print('Generating time-aware aggregate features for fold {} ...'.format(fold))
        features_df = add_fold_aware_features_faster(features_df, train_idx)

        # Specify numeric and categorical features
        features_names = [f for f in features_df.columns if f not in ['date', 'actual', 'on_stock']]
        cat_features = ['product_id', 'product_type_id', 'brand_id', 'manufacturer_id', 'product_group_id', 'team_id', 'subproduct_type_id',
                        'month', 'weekday', 'dayofmonth', 'weekofyear', 'year', 'dayofyear']

        # Split dataframe into train and test set
        train_x, train_y = features_df.iloc[train_idx], target.iloc[train_idx]
        test_x, test_y = features_df.iloc[test_idx], target.iloc[test_idx]

        # Create lgb dataframes and train model
        lgb_train = lgb.Dataset(train_x[features_names], categorical_feature=cat_features, label=train_y, free_raw_data=False)
        lgb_test = lgb.Dataset(test_x[features_names], categorical_feature=cat_features, label=test_y, free_raw_data=False)

        # Clean up memory
        del features_df
        gc.collect()

        # Train LightGBM model
        print('Started training fold {} at {}'.format(fold, datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        print('Train idx: {} to {}, test idx: {} to {}'.format(train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

        booster = lgb.train(
            PARAMS,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_test],
            categorical_feature=cat_features,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        train_preds = booster.predict(lgb_train.data, num_iteration=booster.best_iteration)
        test_preds = booster.predict(lgb_test.data, num_iteration=booster.best_iteration)

        # Clean up memory
        del lgb_train, lgb_test
        gc.collect()

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
            fold_result_df.to_csv('{}.csv'.format(tf.name), index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name),
                               '{}/results_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])

        # Clean up memory
        del fold_result_df
        gc.collect()

        # Compute feature importances
        gain = booster.feature_importance(importance_type='gain')
        split = booster.feature_importance(importance_type='split')
        fold_importance_df = pd.DataFrame()
        fold_importance_df['fold'] = np.repeat(fold, len(features_names))
        fold_importance_df['feature'] = features_names
        fold_importance_df['gain'] = gain
        fold_importance_df['split'] = split

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            fold_importance_df.to_csv('{}.csv'.format(tf.name), index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name),
                               '{}/importance_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])

        # Clean up memory
        del fold_importance_df
        gc.collect()

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
                shap_df.to_csv('{}.csv'.format(tf.name), index=False)
                upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name),
                                   '{}/shap_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
            subprocess.call(['rm', '-f', tf.name])
            subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])

            # Clean up memory
            del shap_df
            gc.collect()

        # Save booster object to disk
        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            booster.save_model('{}.txt'.format(tf.name))
            upload_file_to_gcs(PROJECT, BUCKET, '{}.txt'.format(tf.name),
                               '{}/booster_{}_{}.txt'.format(MODEL_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.txt'.format(tf.name)])

        # Clean up memory
        del train_x, train_y, test_x, test_y
        gc.collect()
