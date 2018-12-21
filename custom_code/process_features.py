from datetime import datetime

from workalendar.europe import Netherlands, Belgium
import pandas as pd
import numpy as np

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
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])

    # Encode different holidays as categories
    # holiday_df[['holiday_nl', 'holiday_be']] = holiday_df[['holiday_nl', 'holiday_be']].fillna(value='None')
    # holiday_df['holiday_nl'] = holiday_df['holiday_nl'].astype('category')
    # holiday_df['holiday_be'] = holiday_df['holiday_be'].astype('category')
    # holiday_df['holiday_nl_cat'] = holiday_df['holiday_nl'].cat.codes
    # holiday_df['holiday_be_cat'] = holiday_df['holiday_be'].cat.codes

    # Black friday (Only 2017 for now, nothing much happened in 2016)
    # holiday_df = holiday_df.append({'date': '2017-11-24', 'holiday_nl': 'Black Friday', 'holiday_be': 'Black Friday'}, ignore_index=True)
    # holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    # holiday_df.tail(3)

    # Encode all holidays as a single holiday dummy
    holiday_df[['holiday_nl', 'holiday_be']] = holiday_df[['holiday_nl', 'holiday_be']].notnull().astype('int8')

    # Merge generated features into dataframe
    data_df = pd.merge(data_df, holiday_df, on='date', how='left')
    del holiday_df

    # Set Non Holidays to 0
    data_df[['holiday_nl', 'holiday_be']] = data_df[['holiday_nl', 'holiday_be']].fillna(value=0).astype('int8')

    # Add shifted holiday features
    data_df = data_df.sort_values(by=['date']).reset_index(drop=True)
    for shift in [1, 2, 3, 4, 5]:
        data_df['holiday_nl_min_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_nl'].shift(-shift).fillna(value=0).astype('int8')
        data_df['holiday_nl_plus_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_nl'].shift(shift).fillna(value=0).astype('int8')
        data_df['holiday_be_min_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_be'].shift(-shift).fillna(value=0).astype('int8')
        data_df['holiday_be_plus_{}'.format(shift)] = data_df.groupby(['product_id'])['holiday_be'].shift(shift).fillna(value=0).astype('int8')

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
            grouped_lag_target_df[column_name].rolling(lag).min().fillna(0).reset_index(0, drop=True)
        LOGGER.info('Creating lag-max feature')
        lag_target_df['{}_max'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).max().fillna(0).reset_index(0, drop=True)
        LOGGER.info('Creating lag-mean feature')
        lag_target_df['{}_mean'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).mean().fillna(0).reset_index(0, drop=True)
        LOGGER.info('Creating lag-median feature')
        lag_target_df['{}_median'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).median().fillna(0).reset_index(0, drop=True)
        LOGGER.info('Creating lag-sum feature')
        lag_target_df['{}_sum'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).sum().fillna(0).reset_index(0, drop=True)
        LOGGER.info('Creating lag-var feature')
        lag_target_df['{}_var'.format(column_name)] = \
            grouped_lag_target_df[column_name].rolling(lag).var().fillna(0).reset_index(0, drop=True)

        downcast_datatypes(lag_target_df)

    lag_target_df = lag_target_df.drop('actual', axis=1)
    # Remove all data for which we don't have accurate lag targets, namely the first max(lag) days per product
    # TODO: IMPLEMENT

    LOGGER.info('Merging data dataframe and lags')
    data_df = pd.merge(data_df, lag_target_df, how='left', on=['product_id', 'date'])
    # With this fillna, we say that, for all out of stock days, lag target and its rolling aggregates is 0
    # If lag target gets a lot of effect, this should help with pushing down the prediction
    # We can change this (maybe the min?) depending on how it comes out


    # LOGGER.info('Filling missing data in columns in merged dataframe')
    # data_df = data_df.fillna(0)

    LOGGER.info('Downcast datatypes in merged dataframe')
    # for lag in lags:
    #     column_name = 'lag_{}'.format(lag)
    #     int16_features = [column_name, '{}_min'.format(column_name), '{}_max'.format(column_name), '{}_median'.format(column_name), '{}_sum'.format(column_name)]
    #     float16_features = ['{}_mean'.format(column_name), '{}_var'.format(column_name)]
    #     data_df[int16_features] = data_df[int16_features].apply(lambda col: col.astype('int16'))
    #     data_df[float16_features] = data_df[float16_features].apply(lambda col: col.astype('float16'))
    downcast_datatypes(data_df)
    return data_df


def add_product_features(data_df):
    print('Generating product based features ...')
    file_location = download_file_from_gcs(PROJECT, BUCKET,
                                           '{}/product_{}.h5'.format(DATA_DIR, 'ALL_PRODUCTS_LESS_FEATURES'))
    product_df = pd.read_hdf(file_location, 'product_df')
    data_df = data_df.merge(product_df, on='product_id', how='inner')
    downcast_datatypes(data_df)
    return data_df


def process_features(data_df, LOGGER):
    data_df['date'] = pd.to_datetime(data_df['date'])

    LOGGER.info('Processing seasonal features')
    data_df = add_seasonal_features(data_df)
    # LOGGER.info('Processing holiday features')
    # data_df = add_holiday_features(data_df)
    LOGGER.info('Processing lag features')
    data_df = add_lag_target_and_rolling_aggregate_features(data_df, lags=LAGS, LOGGER=LOGGER)
    LOGGER.info('Processing product features')
    data_df = add_product_features(data_df)

    LOGGER.info('Sorting features by date')
    data_df = data_df.sort_values(by='date').reset_index(drop=True)
    LOGGER.info('Finish features generation')
    return data_df


def add_fold_aware_features(train_features_df, test_features_df):
    # Compute fold aggregation features (only based on train set -> no data leakage)
    full_num_aggregations = {'actual': ['mean', 'median', 'var']}
    weekday_num_aggregations = {'actual': ['max', 'mean', 'median']}
    # isoweek_num_aggregations = {'actual': ['max', 'mean', 'median', 'sum']}

    full_agg_df = train_features_df.groupby('product_id').agg(full_num_aggregations)
    full_agg_df.columns = ["_full_".join(agg_feature) for agg_feature in full_agg_df.columns.ravel()]
    full_agg_df.reset_index(drop=False, inplace=True)

    weekday_agg_df = train_features_df.groupby(['product_id', 'weekday']).agg(weekday_num_aggregations)
    weekday_agg_df.columns = ["_weekday_".join(agg_feature) for agg_feature in weekday_agg_df.columns.ravel()]
    weekday_agg_df.reset_index(drop=False, inplace=True)

    downcast_datatypes(full_agg_df)
    downcast_datatypes(weekday_agg_df)

    # month_agg_df = df.iloc[train_idx].groupby(['product_id', 'month']).agg(num_aggregations)
    # month_agg_df.columns = ["_month_".join(agg_feature) for agg_feature in month_agg_df.columns.ravel()]
    # month_agg_df.reset_index(drop=False, inplace=True)

    # isoweek_agg_df = df.iloc[train_idx].groupby(['product_id', 'weekofyear']).agg(isoweek_num_aggregations)
    # isoweek_agg_df.columns = ["_isoweek_".join(agg_feature) for agg_feature in isoweek_agg_df.columns.ravel()]
    # isoweek_agg_df.reset_index(drop=False, inplace=True)

    print('Merging aggregation features into dataframe ...')
    # Left join to preserve index of features_df
    train_features_df = pd.merge(train_features_df, full_agg_df, how='left', on='product_id')
    train_features_df = pd.merge(train_features_df, weekday_agg_df, how='left', on=['product_id', 'weekday'])

    test_features_df = pd.merge(test_features_df, full_agg_df, how='left', on='product_id')
    test_features_df = pd.merge(test_features_df, weekday_agg_df, how='left', on=['product_id', 'weekday'])
    # features_df = pd.merge(features_df, month_agg_df, how='inner', on=['product_id', 'month'])
    # features_df = pd.merge(features_df, isoweek_agg_df, how='inner', on=['product_id', 'weekofyear'])

    # Test adding some interactions between important features
    print('Generating interaction features ...')
    train_features_df['full_mean_lag_7_median_prod'] = (train_features_df['actual_full_mean'] * train_features_df['lag_7_median'])
    train_features_df['full_mean_lag_7_mean_prod'] = (train_features_df['actual_full_mean'] * train_features_df['lag_7_mean'])
    train_features_df['weekday_mean_lag_7_median_prod'] = (train_features_df['actual_weekday_mean'] * train_features_df['lag_7_median'])

    test_features_df['full_mean_lag_7_median_prod'] = (test_features_df['actual_full_mean'] * test_features_df['lag_7_median'])
    test_features_df['full_mean_lag_7_mean_prod'] = (test_features_df['actual_full_mean'] * test_features_df['lag_7_mean'])
    test_features_df['weekday_mean_lag_7_median_prod'] = (test_features_df['actual_weekday_mean'] * test_features_df['lag_7_median'])

    downcast_datatypes(train_features_df)
    downcast_datatypes(test_features_df)

    return train_features_df, test_features_df


def add_fold_aware_features_faster(features_df, train_idx):
    # Compute fold aggregation features (only based on train set -> no data leakage)
    full_num_aggregations = {'actual': ['mean', 'median', 'var']}
    weekday_num_aggregations = {'actual': ['max', 'mean', 'median']}
    # isoweek_num_aggregations = {'actual': ['max', 'mean', 'median', 'sum']}

    full_agg_df = features_df.iloc[train_idx].groupby('product_id').agg(full_num_aggregations)
    full_agg_df.columns = ["_full_".join(agg_feature) for agg_feature in full_agg_df.columns.ravel()]
    full_agg_df.reset_index(drop=False, inplace=True)

    weekday_agg_df = features_df.iloc[train_idx].groupby(['product_id', 'weekday']).agg(weekday_num_aggregations)
    weekday_agg_df.columns = ["_weekday_".join(agg_feature) for agg_feature in weekday_agg_df.columns.ravel()]
    weekday_agg_df.reset_index(drop=False, inplace=True)

    downcast_datatypes(full_agg_df)
    downcast_datatypes(weekday_agg_df)

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

    # Test adding some interactions between important features
    print('Generating interaction features ...')
    features_df['full_mean_lag_7_median_prod'] = features_df['actual_full_mean'] * features_df['lag_7_median'].astype('float16')
    features_df['full_mean_lag_7_mean_prod'] = features_df['actual_full_mean'] * features_df['lag_7_mean'].astype('float16')
    features_df['weekday_mean_lag_7_median_prod'] = features_df['actual_weekday_mean'] * features_df['lag_7_median'].astype('float16')

    downcast_datatypes(features_df)

    return features_df
