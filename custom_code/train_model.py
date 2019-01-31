from datetime import datetime
import gc
import subprocess
import tempfile

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

from custom_code.process_features import add_fold_aware_features_faster, downcast_datatypes
from custom_code.upload_file_to_gcs import upload_file_to_gcs
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code import timefold
from custom_code.settings import PROJECT, BUCKET, DATA_DIR, MODEL_DIR, RESULTS_DIR, RUNTAG, COMPUTE_SHAP, PARAMS


def train_model(features_df, LOGGER):

    LOGGER.info('Subsetting data to 40.000 most recent products')
    features_df = features_df[features_df['product_id'].isin(
        features_df['product_id'].tail(40000))].copy() # Subset to 10.000 most recent products
    LOGGER.info('Finished subsetting')

    LOGGER.info('Setting up LightGBM ...')
    target = features_df['actual'].copy()
    feature_importance_df = pd.DataFrame()

    LOGGER.info('Dropping useless features for now')
    drop_features = ['quarter', 'team_id', 'manufacturer_id', 'year', 'brand_id']
    features_df = features_df.drop(drop_features, axis=1).copy()

    LOGGER.info('Downcasting datatypes')
    features_df = downcast_datatypes(features_df)

    if not COMPUTE_SHAP:
        LOGGER.info('Warning: Shapley value computation is disabled')

    # Cross-validation setup using timefold
    min_train_size = int(0.75 * features_df.shape[0])  # >2 years training
    min_test_size = int(0.0075 * features_df.shape[0])  # ~7 days of testing (similar to situation in production)
    step_size = int(0.03 * features_df.shape[0])  # ~1 month step size between folds
    timefolds = timefold.timefold(method='step', min_train_size=min_train_size, min_test_size=min_test_size,
                                  step_size=step_size)

    for fold, (train_idx, test_idx) in enumerate(timefolds.split(features_df)):
            LOGGER.info('Start processing fold {}'.format(fold))
            LOGGER.info('Copy features dataframe in another variable')
            features_df_tmp = features_df.copy()
            LOGGER.info('Train idx: {} to {}, test idx: {} to {}'.format(
                train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

            LOGGER.info('Generating fold-aware aggregate features for fold {}'.format(fold))
            features_df_tmp = add_fold_aware_features_faster(features_df_tmp, train_idx)
            features_df_tmp = downcast_datatypes(features_df_tmp)
            LOGGER.info('Finished generating fold-aware features for fold {}'.format(fold))

            # Specify numeric and categorical features
            features_names = [f for f in features_df_tmp.columns if f not in ['date', 'actual', 'on_stock', 'product_id']]
            cat_features = ['product_type_id', 'product_group_id', 'subproduct_type_id',
                            'month', 'weekday', 'dayofmonth', 'weekofyear', 'dayofyear']

            # Split dataframe into train and test set
            LOGGER.info('Start splitting dataframe')
            train_x, train_y = features_df_tmp.iloc[train_idx], target.iloc[train_idx]
            test_x, test_y = features_df_tmp.iloc[test_idx], target.iloc[test_idx]
            LOGGER.info('Finished splitting dataframe')

            # Create lgb dataframes and train model
            LOGGER.info('Start creating lgb dataframes')
            lgb_train = lgb.Dataset(train_x[features_names],
                                    categorical_feature=cat_features, label=train_y, free_raw_data=False)
            lgb_test = lgb.Dataset(test_x[features_names], categorical_feature=cat_features,
                                   label=test_y, free_raw_data=False)
            LOGGER.info('Finished creating lgb dataframes')

            # Clean up memory
            del features_df_tmp
            gc.collect()

            # Train LightGBM model
            LOGGER.info('Started training fold {}'.format(fold))
            LOGGER.info('Train idx: {} to {}, test idx: {} to {}'.format(
                train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

            booster = lgb.train(
                PARAMS,
                lgb_train,
                num_boost_round=3000,
                valid_sets=[lgb_train, lgb_test],
                categorical_feature=cat_features,
                early_stopping_rounds=100,
                verbose_eval=100
            )
            LOGGER.info('Finished training fold {}'.format(fold))

            LOGGER.info('Predict training with booster')
            train_preds = booster.predict(lgb_train.data, num_iteration=booster.best_iteration)
            LOGGER.info('Predict testing with booster')
            test_preds = booster.predict(lgb_test.data, num_iteration=booster.best_iteration)

            # Clean up memory
            del lgb_train, lgb_test
            gc.collect()

            LOGGER.info('Generating results dataframe for fold {}'.format(fold))
            fold_result_df = pd.DataFrame()
            fold_result_df['product_id'] = train_x['product_id'].append(test_x['product_id'])
            fold_result_df['date'] = train_x['date'].append(test_x['date'])
            fold_result_df['on_stock'] = train_x['on_stock'].append(test_x['on_stock'])
            fold_result_df['fold'] = np.repeat(fold, len(train_x.index) + len(test_x.index))
            fold_result_df['actual'] = train_x['actual'].append(test_x['actual'])
            fold_result_df['lgbm'] = np.concatenate([train_preds, test_preds])
            fold_result_df['is_test'] = np.concatenate(
                [np.repeat(False, len(train_x.index)), np.repeat(True, len(test_x.index))])
            fold_result_df = fold_result_df.sort_values(by=['product_id', 'date'])

            LOGGER.info('Write results to local file')
            fold_result_df.to_csv('./results_{}_{}.csv'.format(fold, RUNTAG), index=False)
            LOGGER.info('Write results to GS bucket')
            upload_file_to_gcs(PROJECT, BUCKET, './results_{}_{}.csv'.format(fold, RUNTAG),
                               '{}/results_{}_{}.csv'.format(RESULTS_DIR, fold, RUNTAG))
            subprocess.call(['rm', '-f', './results_{}_{}.csv'.format(fold, RUNTAG)])
            del fold_result_df
            gc.collect()

            # Compute feature importances
            LOGGER.info('Start computing fold feature importances')
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = features_names
            fold_importance_df['gain'] = booster.feature_importance(importance_type='gain')
            fold_importance_df['split'] = booster.feature_importance(importance_type='split')
            fold_importance_df['fold'] = fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            del fold_importance_df
            gc.collect()

            # Only save objects for SHAP computation if specified
            if COMPUTE_SHAP:
                LOGGER.info('Writing objects for SHAP computations')

                # Save booster object to disk for SHAP plots
                LOGGER.info('Start writing booster object to local file')
                booster.save_model('./booster_{}_{}.txt'.format(fold, RUNTAG))
                LOGGER.info('Start writing booster object to GS bucket')
                upload_file_to_gcs(PROJECT, BUCKET, './booster_{}_{}.txt'.format(fold, RUNTAG),
                                   '{}/booster_{}_{}.txt'.format(MODEL_DIR, fold, RUNTAG))
                subprocess.call(['rm', '-f', './booster_{}_{}.txt'.format(fold, RUNTAG)])

                # Save test matrix to disk for SHAP plots
                LOGGER.info('Start writing test matrix to local file')
                test_x.to_hdf('./test_x_{}_{}.hdf'.format(fold, RUNTAG), 'test_df', index=False)
                LOGGER.info('Start writing test matrix to GS bucket')
                upload_file_to_gcs(PROJECT, BUCKET, './test_x_{}_{}.hdf'.format(fold, RUNTAG),
                                   '{}/test_x_{}_{}.hdf'.format(RESULTS_DIR, fold, RUNTAG))
                subprocess.call(['rm', '-f', './test_x_{}_{}.hdf'.format(fold, RUNTAG)])

            # Clean up memory
            del train_x, train_y, test_x, test_y
            gc.collect()

            LOGGER.info('Finished processing fold {}'.format(fold))

    LOGGER.info('Write overall importances to local file')
    feature_importance_df.to_csv('./overall_importance_{}.csv'.format(RUNTAG), index=False)
    LOGGER.info('Write overall importances to GS bucket')
    upload_file_to_gcs(PROJECT, BUCKET, './overall_importance_{}.csv'.format(RUNTAG),
                       '{}/overall_importance_{}.csv'.format(RESULTS_DIR, RUNTAG))
    subprocess.call(['rm', '-f', './overall_importance_{}.csv'.format(RUNTAG)])
