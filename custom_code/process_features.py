from datetime import datetime

from workalendar.europe import Netherlands, Belgium
import pandas as pd
import numpy as np
from numpy import nan, inf
import gc

from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.settings import LAGS, PROJECT, BUCKET, DATA_DIR, RUNTAG


def downcast_datatypes(df):
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int'])

    for cols in float_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='float')
    for cols in int_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='integer')

    return df


def add_seasonal_features(data_df):
    print('Generating seasonality features ...')
    data_df['year'] = data_df['date'].dt.year.astype('int16')
    data_df['quarter'] = data_df['date'].dt.quarter.astype('int8')
    data_df['month'] = data_df['date'].dt.month.astype('int8')
    data_df['weekday'] = data_df['date'].dt.weekday.astype('int8')
    data_df['dayofmonth'] = data_df['date'].dt.day.astype('int8')
    data_df['weekofyear'] = data_df['date'].dt.week.astype('int8')
    data_df['dayofyear'] = data_df['date'].dt.dayofyear.astype('int16')

    return data_df


def add_holiday_features(data_df):
    # Generate holiday / event features
    print('Generating holiday & event features ...')
    holiday_nl = Netherlands()  # Create holiday objects from workalendar
    holiday_be = Belgium()

    # Initialize lists to fill
    years_in_df = np.unique(data_df['year'])
    holiday_arr_nl = []
    holiday_arr_be = []

    for year in years_in_df:  # Generate holidays based on years present in the data
        holiday_arr_nl.append(holiday_nl.holidays(year))
        holiday_arr_be.append(holiday_be.holidays(year))

    # Combine generated holidays per year into one frame
    holiday_df_nl = pd.DataFrame(holiday_arr_nl[0]).append(pd.DataFrame(holiday_arr_nl[1]))
    holiday_df_nl = holiday_df_nl.append(pd.DataFrame(holiday_arr_nl[2]))
    holiday_df_be = pd.DataFrame(holiday_arr_be[0]).append(pd.DataFrame(holiday_arr_be[1]))
    holiday_df_be = holiday_df_be.append(pd.DataFrame(holiday_arr_be[2]))

    # Set columns
    holiday_df_nl.columns = ['date', 'holiday_nl']
    holiday_df_be.columns = ['date', 'holiday_be']
    # Combine NL and BE holidays
    holiday_df = pd.merge(holiday_df_nl, holiday_df_be, on=['date'], how='outer')

    # Encode different holidays as categories
    # holiday_df[['holiday_nl', 'holiday_be']] = holiday_df[['holiday_nl', 'holiday_be']].fillna(value='None')
    # holiday_df['holiday_nl'] = holiday_df['holiday_nl'].astype('category')
    # holiday_df['holiday_be'] = holiday_df['holiday_be'].astype('category')
    # holiday_df['holiday_nl_cat'] = holiday_df['holiday_nl'].cat.codes
    # holiday_df['holiday_be_cat'] = holiday_df['holiday_be'].cat.codes

    # Black friday (nothing much happened in 2016)
    holiday_df = holiday_df.append({'date': '2017-11-24', 'holiday_nl': 'Black Friday', 'holiday_be': 'Black Friday'}, ignore_index=True)
    holiday_df = holiday_df.append({'date': '2018-11-23', 'holiday_nl': 'Black Friday', 'holiday_be': 'Black Friday'}, ignore_index=True)

    # Encode all holidays as a single holiday dummy
    holiday_df[['holiday_nl', 'holiday_be']] = holiday_df[['holiday_nl', 'holiday_be']].notnull().astype('int8')

    # Merge generated features into dataframe
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    data_df = pd.merge(data_df, holiday_df, on='date', how='left')

    # Set Non Holidays to 0
    data_df[['holiday_nl', 'holiday_be']] = data_df[['holiday_nl', 'holiday_be']].fillna(value=0).astype('int8')

    # Add shifted holiday features
    # data_df = data_df.sort_values(by=['date']).reset_index(drop=True)
    # for shift in [1, 2, 3, 4, 5]:
    #     data_df['holiday_nl_min_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_nl'].shift(-shift).fillna(value=0).astype('int8')
    #     data_df['holiday_nl_plus_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_nl'].shift(shift).fillna(value=0).astype('int8')
    #     data_df['holiday_be_min_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_be'].shift(-shift).fillna(value=0).astype('int8')
    #     data_df['holiday_be_plus_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_be'].shift(shift).fillna(value=0).astype('int8')

    del holiday_df
    gc.collect()

    return data_df


def add_lag_target_and_rolling_aggregate_features(data_df, lags, LOGGER):
    print('Generating lag target features ...')
    # data_df = data_df.sort_values(by='date', ascending=True).reset_index(drop=True)
    # It's very important how we deal with out of stock for these features. We want to shift without taking any out of
    # stock into account, so that we can see the effect of actuals on followup periods
    # But we do want to keep out of stock in the dataframe so that we can predict on those days as well later
    # So we make a separate dataframe that does not contain OOS, generate the lag target features there,
    # and left join it back on the main dataframe
    lag_target_df = data_df[data_df.on_stock][['product_id', 'date', 'actual']]
    LOGGER.info('Sorting data dataframe before doing lags')
    lag_target_df = lag_target_df.sort_values(by='date', ascending=True).reset_index(drop=True)

    for lag in lags:
        LOGGER.info('Generating for lag {} ...'.format(lag))
        column_name = 'lag_{}'.format(lag)
        LOGGER.info('Creating lag feature')
        lag_target_df[column_name] = lag_target_df.groupby('product_id')['actual'].shift(lag).fillna(0).astype('int16')
        grouped_lag_target_df = lag_target_df.groupby('product_id')
        # Generate all rolling aggregates based on the lag target feature, otherwise we'd have leakage
        LOGGER.info('Creating lag-min feature')
        lag_target_df['{}_min'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).min().fillna(0).astype('int16').reset_index(0, drop=True)
        LOGGER.info('Creating lag-max feature')
        lag_target_df['{}_max'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).max().fillna(0).astype('int16').reset_index(0, drop=True)
        LOGGER.info('Creating lag-mean feature')
        lag_target_df['{}_mean'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).mean().fillna(0).astype('float16').reset_index(0, drop=True)
        LOGGER.info('Creating lag-median feature')
        lag_target_df['{}_median'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).median().fillna(0).astype('int16').reset_index(0, drop=True)
        LOGGER.info('Creating lag-sum feature')
        lag_target_df['{}_sum'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).sum().fillna(0).astype('int16').reset_index(0, drop=True)
        LOGGER.info('Creating lag-var feature')
        lag_target_df['{}_var'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).var().fillna(0).astype('float16').reset_index(0, drop=True)

        lag_target_df = downcast_datatypes(lag_target_df)

    lag_target_df = lag_target_df.drop('actual', axis=1)
    # Remove all data for which we don't have accurate lag targets, namely the first max(lag) days per product
    # TODO: IMPLEMENT

    LOGGER.info('Merging data dataframe and lags')
    data_df = pd.merge(data_df, lag_target_df, how='left', on=['product_id', 'date'])
    # With this fillna, we say that, for all out of stock days, lag target and its rolling aggregates is 0
    # If lag target gets a lot of effect, this should help with pushing down the prediction
    # We can change this (maybe the min?) depending on how it comes out
    LOGGER.info('Filling missing data in columns in merged dataframe')
    data_df = data_df.fillna(0)

    LOGGER.info('Setting datatypes in merged dataframe')
    for lag in lags:
        column_name = 'lag_{}'.format(lag)
        int16_features = [column_name, '{}_min'.format(column_name), '{}_max'.format(column_name), '{}_median'.format(column_name), '{}_sum'.format(column_name)]
        float16_features = ['{}_mean'.format(column_name), '{}_var'.format(column_name)]
        data_df[int16_features] = data_df[int16_features].apply(lambda col: col.astype('int16'))
        data_df[float16_features] = data_df[float16_features].apply(lambda col: col.astype('float16'))

    del lag_target_df, grouped_lag_target_df
    gc.collect()

    return data_df


# def add_lag_session_and_rolling_aggregate_features(data_df, lags, LOGGER):
#     LOGGER.info('Reading sessions data with enrichment productid counts')
#     file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/sessions_df.h5'.format(DATA_DIR))
#     sessions_df = pd.read_hdf(file_location, 'sessions_df')
#     LOGGER.info('Generating dataframe to create lag session feature')
#     data_df = data_df.merge(sessions_df, how='left', on=['date', 'product_id']).fillna(0)
#     del sessions_df
#     gc.collect()
#     sessions_df = data_df[data_df.on_stock][['product_id', 'date', 'enrichment_productid_count']]
#     data_df = data_df.drop('enrichment_productid_count', axis=1)
#
#     LOGGER.info('Sorting data dataframe before doing lags')
#     sessions_df = sessions_df.sort_values(by='date', ascending=True).reset_index(drop=True)
#
#     for lag in [7, 8, 9]:
#         LOGGER.info('Generating for sessions lag {} ...'.format(lag))
#         column_name = 'productid_count_lag_{}'.format(lag)
#         LOGGER.info('Creating sessions lag feature')
#         sessions_df[column_name] = sessions_df.groupby('product_id')['enrichment_productid_count'].shift(lag).fillna(0).astype('int16')
#         grouped_sessions_df = sessions_df.groupby('product_id')
#         # # Generate all rolling aggregates based on the lag target feature, otherwise we'd have leakage
#         LOGGER.info('Creating sessions lag-max feature')
#         sessions_df['{}_max'.format(column_name)] = grouped_sessions_df[column_name].rolling(lag).max().fillna(0).astype('int16').reset_index(0, drop=True)
#         # LOGGER.info('Creating sessions lag-median feature')
#         # sessions_df['{}_median'.format(column_name)] = grouped_sessions_df[column_name].rolling(lag).median().fillna(0).astype('int16').reset_index(0, drop=True)
#         LOGGER.info('Creating sessions lag-sum feature')
#         sessions_df['{}_sum'.format(column_name)] = grouped_sessions_df[column_name].rolling(lag).sum().fillna(0).astype('int16').reset_index(0, drop=True)
#
#         sessions_df = downcast_datatypes(sessions_df)
#
#     LOGGER.info('Adding sessions data to feature matrix')
#     sessions_df = sessions_df.drop('enrichment_productid_count', axis=1)
#     data_df = data_df.merge(sessions_df, how='left', on=['date', 'product_id'])
#
#     del sessions_df
#     gc.collect()
#
#     return data_df


# def add_marketing_features(data_df, lags, LOGGER): # This does not run yet, needs work
#     LOGGER.info('Loading marketing features from GCS') # Added
#     file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/marketing_df.h5'.format(DATA_DIR))
#     marketing_df = pd.read_hdf(file_location, 'marketing_df')
#     LOGGER.info('Merging marketing features')
#     data_df = data_df.merge(marketing_df, how='left', on=['product_type_id', 'date'])
#     del marketing_df
#     gc.collect()
#
#     marketing_df = data_df[data_df.on_stock][['product_id', 'date', 'enrichment_productid_count']]
#
#     LOGGER.info('Sorting data dataframe before doing lags')
#     marketing_df = marketing_df.sort_values(by='date', ascending=True).reset_index(drop=True)
#
#     for lag in [7, 8, 9]:
#         LOGGER.info('Generating marketing lag {} ...'.format(lag))
#         column_name = 'productid_count_lag_{}'.format(lag)
#         LOGGER.info('Creating marketing lag feature')
#         marketing_df[column_name] = marketing_df.groupby('product_id')['enrichment_productid_count'].shift(lag).fillna(0).astype('int16')
#         grouped_marketing_df = marketing_df.groupby('product_id')
#         # Generate all rolling aggregates based on the lag target feature, otherwise we'd have leakage
#         LOGGER.info('Creating sessions lag-max feature')
#         marketing_df['{}_max'.format(column_name)] = grouped_marketing_df[column_name].rolling(lag).max().fillna(0).astype('int16').reset_index(0, drop=True)
#         LOGGER.info('Creating sessions lag-median feature')
#         marketing_df['{}_median'.format(column_name)] = grouped_marketing_df[column_name].rolling(lag).median().fillna(0).astype('int16').reset_index(0, drop=True)
#         LOGGER.info('Creating sessions lag-sum feature')
#         marketing_df['{}_sum'.format(column_name)] = grouped_marketing_df[column_name].rolling(lag).sum().fillna(0).astype('int16').reset_index(0, drop=True)
#
#     marketing_df = downcast_datatypes(marketing_df)
#     LOGGER.info('Adding marketing data to feature matrix')
#     marketing_df = marketing_df.drop('enrichment_productid_count', axis=1)
#     data_df = data_df.merge(marketing_df, how='left', on=['date', 'product_id'])
#     del marketing_df
#     LOGGER.info('Returning dataframe with lagged marketing features')
#
#     return data_df


def add_wa_feature(data_df, LOGGER):
    LOGGER.info('Loading WA forecast from GCS')
    file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/wa.h5'.format(DATA_DIR))
    wa_df = pd.read_hdf(file_location, 'wa_forecast_df')
    wa_df = downcast_datatypes(wa_df)
    wa_df['date'] = pd.to_datetime(wa_df['date'])
    LOGGER.info('Adding WA forecast as feature')
    data_df = data_df.merge(wa_df, how='left', on=['product_id', 'date'])

    lag_wa_df = data_df[data_df.on_stock][['product_id', 'date', 'wa']] # Create dataframe to lag wa feature
    data_df = data_df.drop('wa', axis=1) # Drop wa feature again as it contains 1 day ahead information
    LOGGER.info('Sorting data dataframe before doing lags')
    lag_wa_df = lag_wa_df.sort_values(by='date', ascending=True).reset_index(drop=True)

    for lag in [7]:
        LOGGER.info('Generating for lag {} ...'.format(lag))
        column_name = 'wa_lag_{}'.format(lag)
        LOGGER.info('Creating lag feature')
        lag_wa_df[column_name] = lag_wa_df.groupby('product_id')['wa'].shift(lag).fillna(0).astype('int16')
        lag_wa_df = downcast_datatypes(lag_wa_df)

    lag_wa_df = lag_wa_df.drop('wa', axis=1) # Drop the non-lagged wa feature before merging
    LOGGER.info('Merging data dataframe and wa lag')
    data_df = pd.merge(data_df, lag_wa_df, how='left', on=['product_id', 'date'])

    del wa_df, lag_wa_df
    gc.collect()

    return data_df


def add_product_features(data_df, LOGGER):
    LOGGER.info('Generating product based features')
    file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/product.h5'.format(DATA_DIR))
    product_df = pd.read_hdf(file_location, 'product_df')
    data_df = data_df.merge(product_df, on='product_id', how='inner')
    int16_features = ['actual', 'team_id', 'subproduct_type_id']
    int32_features = ['product_id', 'product_type_id', 'brand_id', 'manufacturer_id', 'product_group_id']
    data_df[int16_features] = data_df[int16_features].apply(lambda col: col.astype('int16'))
    data_df[int32_features] = data_df[int32_features].apply(lambda col: col.astype('int32'))

    del product_df
    gc.collect()

    return data_df


def process_features(data_df, LOGGER):
    data_df['date'] = pd.to_datetime(data_df['date'])

    LOGGER.info('Processing seasonal features')
    data_df = add_seasonal_features(data_df)
    LOGGER.info('Processing holiday features')
    data_df = add_holiday_features(data_df)
    LOGGER.info('Processing lag features')
    data_df = add_lag_target_and_rolling_aggregate_features(data_df, lags=LAGS, LOGGER=LOGGER)
    # LOGGER.info('Processing session features')
    # data_df = add_lag_session_and_rolling_aggregate_features(data_df, lags=LAGS, LOGGER=LOGGER)
    # LOGGER.info('Processing marketing features')
    # data_df = add_marketing_features(data_df)
    LOGGER.info('Processing wa feature')
    data_df = add_wa_feature(data_df, LOGGER=LOGGER)
    LOGGER.info('Processing product features')
    data_df = add_product_features(data_df, LOGGER=LOGGER)
    LOGGER.info('Sorting features by date')
    data_df = data_df.sort_values(by='date').reset_index(drop=True)
    LOGGER.info('Finished features generation')
    return data_df


# Functions below are called in train_model as they are depdendent on the cross-validation setup
def add_fold_aware_features_faster(features_df, train_idx):
    # Compute fold aggregation features (only based on train set -> no data leakage)
    full_num_aggregations = {'actual': ['mean', 'median', 'var']}
    weekday_num_aggregations = {'actual': ['max', 'median', 'mean']}
    # isoweek_num_aggregations = {'actual': ['max', 'mean', 'median', 'sum']}

    full_agg_df = features_df.iloc[train_idx].groupby('product_id').agg(full_num_aggregations)
    full_agg_df.columns = ["_full_".join(agg_feature) for agg_feature in full_agg_df.columns.ravel()]
    full_agg_df.reset_index(drop=False, inplace=True)
    int16_features = ['actual_full_median']
    float16_features = ['actual_full_mean', 'actual_full_var']
    full_agg_df[int16_features] = full_agg_df[int16_features].apply(lambda col: col.astype('int16'))
    full_agg_df[float16_features] = full_agg_df[float16_features].apply(lambda col: col.astype('float16'))

    weekday_agg_df = features_df.iloc[train_idx].groupby(['product_id', 'weekday']).agg(weekday_num_aggregations)
    weekday_agg_df.columns = ["_weekday_".join(agg_feature) for agg_feature in weekday_agg_df.columns.ravel()]
    weekday_agg_df.reset_index(drop=False, inplace=True)
    int16_features = ['actual_weekday_max', 'actual_weekday_median']
    float16_features = ['actual_weekday_mean']
    weekday_agg_df[int16_features] = weekday_agg_df[int16_features].apply(lambda col: col.astype('int16'))
    weekday_agg_df[float16_features] = weekday_agg_df[float16_features].apply(lambda col: col.astype('float16'))

    # month_agg_df = df.iloc[train_idx].groupby(['product_id', 'month']).agg(num_aggregations)
    # month_agg_df.columns = ["_month_".join(agg_feature) for agg_feature in month_agg_df.columns.ravel()]
    # month_agg_df.reset_index(drop=False, inplace=True)

    # isoweek_agg_df = df.iloc[train_idx].groupby(['product_id', 'weekofyear']).agg(isoweek_num_aggregations)
    # isoweek_agg_df.columns = ["_isoweek_".join(agg_feature) for agg_feature in isoweek_agg_df.columns.ravel()]
    # isoweek_agg_df.reset_index(drop=False, inplace=True)

    print('Merging aggregation features into dataframe ...')
    # Left join to preserve index of features_df
    features_df = pd.merge(features_df, full_agg_df, how='left', on='product_id')
    features_df = pd.merge(features_df, weekday_agg_df, how='left', on=['product_id', 'weekday'])
    # features_df = pd.merge(features_df, month_agg_df, how='inner', on=['product_id', 'month'])
    # features_df = pd.merge(features_df, isoweek_agg_df, how='inner', on=['product_id', 'weekofyear'])

    # Adding interactions based on most important features
    print('Generating interaction features ...')
    features_df['full_mean_lag_7_mean_prod'] = features_df['actual_full_mean'] * features_df['lag_7_mean'].astype('float16')
    features_df['weekday_mean_lag_7_median_prod'] = features_df['actual_weekday_mean'] * features_df['lag_7_median'].astype('float16')
    features_df['weekday_mean_lag_7_max_prod'] = features_df['actual_weekday_mean'] * features_df['lag_7_max'].astype('float16')

    features_df['full_mean_lag_7_prod'] = features_df['actual_full_mean'] * features_df['lag_7'].astype('float16')
    features_df['weekday_mean_lag_7_prod'] = features_df['actual_weekday_mean'] * features_df['lag_7'].astype('float16')

    features_df['full_mean_lag_wa_prod'] = features_df['actual_full_mean'] * features_df['wa_lag_7'].astype('float16')
    features_df['weekday_mean_lag_wa_prod'] = features_df['actual_weekday_mean'] * features_df['wa_lag_7'].astype('float16')

    del full_agg_df, weekday_agg_df
    gc.collect()

    return features_df


# def add_experimental_features(features_df, train_idx):
#     # Compute fold aggregation features (only based on train set -> no data leakage)
#
#     # Define dictionary to control feature generation settings
#     print('Defining dictionary with experimental features settings ...')
#     settings = {'variance_larger_than_standard_deviation': None,
#                 'has_duplicate_max': None,
#                 'has_duplicate_min': None,
#                 'sum_values': None,
#                 'abs_energy': None,
#                 'mean_abs_change': None,
#                 'mean_change': None,
#                 'mean_second_derivative_central': None,
#                 'mean': None,
#                 'length': None,
#                 'standard_deviation': None,
#                 'variance': None,
#                 'skewness': None,
#                 'kurtosis': None,
#                 'absolute_sum_of_changes': None,
#                 'longest_strike_below_mean': None,
#                 'longest_strike_above_mean': None,
#                 'count_above_mean': None,
#                 'count_below_mean': None,
#                 'last_location_of_maximum': None,
#                 'first_location_of_maximum': None,
#                 'last_location_of_minimum': None,
#                 'first_location_of_minimum': None,
#                 'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
#                 'percentage_of_reoccurring_values_to_all_values': None,
#                 'sum_of_reoccurring_values': None,
#                 'sum_of_reoccurring_data_points': None,
#                 'ratio_value_number_to_time_series_length': None,
#                 'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
#                 'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
#                 'cid_ce': [{'normalize': True}, {'normalize': False}],
#                 'symmetry_looking': [{'r': 0.0},
#                  {'r': 0.05},
#                  {'r': 0.1},
#                  {'r': 0.2},
#                  {'r': 0.25},
#                  {'r': 0.5},
#                  {'r': 0.75},
#                  {'r': 0.9}],
#                 'large_standard_deviation': [{'r': 0.05},
#                  {'r': 0.25},
#                  {'r': 0.5},
#                  {'r': 0.75},
#                  {'r': 0.9500000000000001}],
#                 'quantile': [{'q': 0.1},
#                  {'q': 0.2},
#                  {'q': 0.3},
#                  {'q': 0.4},
#                  {'q': 0.6},
#                  {'q': 0.7},
#                  {'q': 0.8},
#                  {'q': 0.9}],
#                 'autocorrelation': [{'lag': 0},
#                  {'lag': 1},
#                  {'lag': 2},
#                  {'lag': 3},
#                  {'lag': 4},
#                  {'lag': 5},
#                  {'lag': 6},
#                  {'lag': 7},
#                  {'lag': 8},
#                  {'lag': 9}],
#                 'agg_autocorrelation': [{'f_agg': 'mean'},
#                  {'f_agg': 'median'},
#                  {'f_agg': 'var'}],
#                 'partial_autocorrelation': [{'lag': 0},
#                  {'lag': 1},
#                  {'lag': 2},
#                  {'lag': 3},
#                  {'lag': 4},
#                  {'lag': 5},
#                  {'lag': 6},
#                  {'lag': 7},
#                  {'lag': 8},
#                  {'lag': 9}],
#                 'number_cwt_peaks': [{'n': 1}, {'n': 5}],
#                 'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}, {'n': 50}],
#                 'binned_entropy': [{'max_bins': 10}],
#                 'index_mass_quantile': [{'q': 0.1},
#                  {'q': 0.2},
#                  {'q': 0.3},
#                  {'q': 0.4},
#                  {'q': 0.6},
#                  {'q': 0.7},
#                  {'q': 0.8},
#                  {'q': 0.9}],
#                 'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5},
#                  {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 10},
#                  {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 5},
#                  {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 10},
#                  {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 5},
#                  {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 10},
#                  {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 5},
#                  {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 5},
#                  {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 10},
#                  {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 20},
#                  {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 2},
#                  {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 20}],
#                 'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
#                 'ar_coefficient': [{'coeff': 0, 'k': 10},
#                  {'coeff': 1, 'k': 10},
#                  {'coeff': 2, 'k': 10},
#                  {'coeff': 3, 'k': 10},
#                  {'coeff': 4, 'k': 10}],
#                 'change_quantiles': [{'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.2, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.6, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.6, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.6, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.8, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.8, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.8, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.8, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.8, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
#                  {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
#                  {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
#                  {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
#                  {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}],
#                 'fft_coefficient': [{'coeff': 0, 'attr': 'real'},
#                  {'coeff': 1, 'attr': 'real'},
#                  {'coeff': 25, 'attr': 'real'},
#                  {'coeff': 50, 'attr': 'real'},
#                  {'coeff': 75, 'attr': 'real'},
#                  {'coeff': 99, 'attr': 'real'},
#                  {'coeff': 0, 'attr': 'imag'},
#                  {'coeff': 1, 'attr': 'imag'},
#                  {'coeff': 25, 'attr': 'imag'},
#                  {'coeff': 50, 'attr': 'imag'},
#                  {'coeff': 75, 'attr': 'imag'},
#                  {'coeff': 99, 'attr': 'imag'},
#                  {'coeff': 0, 'attr': 'abs'},
#                  {'coeff': 1, 'attr': 'abs'},
#                  {'coeff': 25, 'attr': 'abs'},
#                  {'coeff': 50, 'attr': 'abs'},
#                  {'coeff': 75, 'attr': 'abs'},
#                  {'coeff': 99, 'attr': 'abs'},
#                  {'coeff': 0, 'attr': 'angle'},
#                  {'coeff': 1, 'attr': 'angle'},
#                  {'coeff': 25, 'attr': 'angle'},
#                  {'coeff': 50, 'attr': 'angle'},
#                  {'coeff': 75, 'attr': 'angle'},
#                  {'coeff': 99, 'attr': 'angle'}],
#                 'fft_aggregated': [{'aggtype': 'centroid'},
#                  {'aggtype': 'variance'},
#                  {'aggtype': 'skew'},
#                  {'aggtype': 'kurtosis'}],
#                 'value_count': [{'value': 0},
#                  {'value': 1},
#                  {'value': nan},
#                  {'value': inf},
#                  {'value': -inf}],
#                 'range_count': [{'min': -1, 'max': 1}],
#                 'friedrich_coefficients': [{'coeff': 0, 'm': 3, 'r': 30},
#                  {'coeff': 1, 'm': 3, 'r': 30},
#                  {'coeff': 2, 'm': 3, 'r': 30},
#                  {'coeff': 3, 'm': 3, 'r': 30}],
#                 'max_langevin_fixed_point': [{'m': 3, 'r': 30}],
#                 'linear_trend': [{'attr': 'pvalue'},
#                  {'attr': 'rvalue'},
#                  {'attr': 'intercept'},
#                  {'attr': 'slope'},
#                  {'attr': 'stderr'}],
#                 'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'},
#                  {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'min'},
#                  {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'},
#                  {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'},
#                  {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'max'},
#                  {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'},
#                  {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'},
#                  {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'},
#                  {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'},
#                  {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'min'},
#                  {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'mean'},
#                  {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'var'},
#                  {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'},
#                  {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'},
#                  {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'},
#                  {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'},
#                  {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'},
#                  {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'},
#                  {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'},
#                  {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'},
#                  {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'},
#                  {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'},
#                  {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'},
#                  {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'},
#                  {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'max'},
#                  {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'min'},
#                  {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'},
#                  {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'var'},
#                  {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'max'},
#                  {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'min'},
#                  {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'mean'},
#                  {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'},
#                  {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'},
#                  {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'min'},
#                  {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'},
#                  {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'},
#                  {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'},
#                  {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'min'},
#                  {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
#                  {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'},
#                  {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'},
#                  {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'},
#                  {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'},
#                  {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'var'},
#                  {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'max'},
#                  {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'},
#                  {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'},
#                  {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'var'}],
#                 'augmented_dickey_fuller': [{'attr': 'teststat'},
#                  {'attr': 'pvalue'},
#                  {'attr': 'usedlag'}],
#                 'number_crossing_m': [{'m': 0}, {'m': -1}, {'m': 1}],
#                 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0},
#                  {'num_segments': 10, 'segment_focus': 1},
#                  {'num_segments': 10, 'segment_focus': 3},
#                  {'num_segments': 10, 'segment_focus': 5},
#                  {'num_segments': 10, 'segment_focus': 9}],
#                 'ratio_beyond_r_sigma': [{'r': 0.5},
#                  {'r': 1},
#                  {'r': 2},
#                  {'r': 3},
#                  {'r': 7},
#                  {'r': 10}]}
#
#     print('Generating experimental features ...')
#     masked_features_df = features_df[['product_id', 'date', 'actual']].iloc[train_idx]
#     experimental_features_df = tsfresh.extract_features(masked_features_df, column_id='product_id', column_sort='date', column_value='actual',
#                                           default_fc_parameters=settings)  # Extract features
#
#     print('Merging experimental features into dataframe ...')
#     experimental_features_df.reset_index(drop=False)
#     features_df = features_df.merge(experimental_features_df, how='left', left_on=['product_id'], right_on=['id'])
#
#     del masked_features_df, experimental_features_df
#
#     return features_df
