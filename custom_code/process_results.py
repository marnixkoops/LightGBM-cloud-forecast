import subprocess
import gc
from sklearn import metrics
import pandas as pd
import numpy as np
import dask.dataframe as dd

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

from custom_code.metrics import mean_huber, mape, wmape
from custom_code.plotting import plot_importances
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.settings import RUNTAG, COMPUTE_SHAP, PROJECT, BUCKET, RESULTS_DIR, DATA_DIR


def downcast_datatypes(df):
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int'])

    for cols in float_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='float')
    for cols in int_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='integer')

    return df


def process_results(results_df, features_importance_df, params):
    print('Reading WA data ...')
    # Join with baseline model
    file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/wa.h5'.format(DATA_DIR))
    wa_df = pd.read_hdf(file_location, 'wa_forecast_df')
    wa_df = downcast_datatypes(wa_df)
    results_df = downcast_datatypes(results_df)
    results_df = pd.merge(results_df, wa_df, how='left', on=['date', 'product_id'])
    subprocess.call(['rm', '-f', file_location])
    del wa_df

    print('Reading product data ...')
    file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/product.h5'.format(DATA_DIR))
    product_df = pd.read_hdf(file_location, 'product_df')
    results_df = pd.merge(results_df, product_df[['product_id', 'product_type_id']], how='left', on='product_id')
    subprocess.call(['rm', '-f', file_location])
    del product_df

    # Create a DF that only covers the test periods
    # We may not have WA for all dates within test, so drop those that don't
    # Daily updated forecast is transformed to a 7-day ahead forecast starting from each first day in each testing fold
    test_df = results_df[results_df['is_test'] == True]

    # Using test_df.iloc[::7, ] to select every 7th row (including 0) inside a group may offer a faster solution
    def transform_to_weekly_wa(test_df):
        """
        Select all rows in the test set that are not divisible by 7 (these are the first days of the weeks)
        Set all but the first days of the week to 0, then fill all NaN's with a forwardfill
        This propogates the first forecasted value of the week to all consecutive days in that week
        The end result is a fair 7-day ahead forecast for each week which is updated every 7 days
        """
        test_df['wa'][test_df.reset_index(drop=True).index % 7 != 0] = np.NaN
        test_df.fillna(method='ffill', inplace=True)
        return test_df

    print('Transforming WA forecast to weekly predictions ...')
    # Ensure to not propogate values across different folds for a product by also grouping by fold!
    test_df = test_df.groupby(['product_id', 'fold']).apply(transform_to_weekly_wa)
    test_df = test_df.dropna()
    test_df = downcast_datatypes(test_df)

    features_names = features_importance_df['feature'].unique()  # get features for log (unique because n folds in df)
    write_metrics(test_df, 'lgbm_log.txt', features_names, params)

    # Plotting
    print('Saving feature importance plots to GCS ...')
    overall_huber_lgbm = mean_huber(test_df['actual'], test_df['lgbm'])
    plot_importances(features_importance_df, overall_huber_lgbm, type='split')
    plot_importances(features_importance_df, overall_huber_lgbm, type='gain')
    del features_importance_df


def write_metrics(test_df, filepath, features_names, params):
    print('Calculating overall and product group metrics ...')

    overall_huber_lgbm = mean_huber(test_df['actual'], test_df['lgbm'])
    overall_mse_lgbm = metrics.mean_squared_error(test_df['actual'], test_df['lgbm'])
    overall_mae_lgbm = metrics.mean_absolute_error(test_df['actual'], test_df['lgbm'])
    overall_mape_lgbm = mape(test_df['actual'], test_df['lgbm'])
    overall_wmape_lgbm = wmape(test_df['actual'], test_df['lgbm'])

    overall_huber_wa = mean_huber(test_df['actual'], test_df['wa'])
    overall_mse_wa = metrics.mean_squared_error(test_df['actual'], test_df['wa'])
    overall_mae_wa = metrics.mean_absolute_error(test_df['actual'], test_df['wa'])
    overall_mape_wa = mape(test_df['actual'], test_df['wa'])
    overall_wmape_wa = wmape(test_df['actual'], test_df['wa'])

    test_df_oos = test_df[test_df['product_type_id'].isin([4396, 4999, 5539, 2369, 2703])].dropna()
    test_df_promo = test_df[test_df['product_type_id'].isin([2233, 2341, 2627, 5600, 2063])].dropna()
    test_df_normal = test_df[test_df['product_type_id'].isin([2452, 2458, 2096, 2090,
                                                              2048, 2250, 2562, 9504])].dropna()

    print('Overall Huber LGBM: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
        overall_huber_lgbm, overall_mse_lgbm, overall_mae_lgbm, overall_mape_lgbm, overall_wmape_lgbm))

    print('Overall Huber WA: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
        overall_huber_wa, overall_mse_wa, overall_mae_wa, overall_mape_wa, overall_wmape_wa))

    model_id = RUNTAG + '_Huber_{:.5}'.format(overall_huber_lgbm)

    # Write model results to local log
    print('Writing model info and results to log ...')
    with open(filepath, "a+") as text_file:
        print(
            "[----------------------------------------------------------------------------------------------------------]",
            file=text_file)
        print("[+] Model ID: {}".format(model_id), file=text_file)
        print("Test Range: from {} to {}".format(test_df['date'].min(), test_df['date'].max()), file=text_file)

        print('[+] LGBM', file=text_file)
        print('Overall Huber: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
            overall_huber_lgbm, overall_mse_lgbm, overall_mae_lgbm, overall_mape_lgbm, overall_wmape_lgbm), file=text_file)

        print('[+] WA', file=text_file)
        print('Overall Huber: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
            overall_huber_wa, overall_mse_wa, overall_mae_wa, overall_mape_wa, overall_wmape_wa), file=text_file)
        print("[+] Parameters: {}".format(params), file=text_file)
        print("[+] Features: {}".format(features_names), file=text_file)

        n_folds = max(test_df['fold'])
        for fold in range(n_folds + 1):
            fold_test_df = test_df[test_df.fold == fold]
            print('[+] Fold {} from {} to {}'.format(fold, fold_test_df['date'].min(), fold_test_df['date'].max()),
                  file=text_file)

            print('Huber\t| LightGBM: {:.5}\t\tWeighted Average: {:.5}'.format(
                mean_huber(fold_test_df['actual'], fold_test_df['lgbm']),
                mean_huber(fold_test_df['actual'], fold_test_df['wa'])), file=text_file)

            print('MSE\t\t| LightGBM: {:.5}\t\tWeighted Average: {:.5}'.format(
                metrics.mean_squared_error(fold_test_df['actual'], fold_test_df['lgbm']),
                metrics.mean_squared_error(fold_test_df['actual'], fold_test_df['wa'])), file=text_file)

            print('MAE\t\t| LightGBM: {:.5}\t\tWeighted Average: {:.5}'.format(
                metrics.mean_absolute_error(fold_test_df['actual'], fold_test_df['lgbm']),
                metrics.mean_absolute_error(fold_test_df['actual'], fold_test_df['wa'])), file=text_file)

            print('MAPE\t| LightGBM: {:.5}\t\tWeighted Average: {:.5}'.format(
                mape(fold_test_df['actual'], fold_test_df['lgbm']),
                mape(fold_test_df['actual'], fold_test_df['wa'])), file=text_file)

            print('WMAPE\t| LightGBM: {:.5}\t\tWeighted Average: {:.5}'.format(
                wmape(fold_test_df['actual'], fold_test_df['lgbm']),
                wmape(fold_test_df['actual'], fold_test_df['wa'])), file=text_file)

        print("[+] Product Group Metrics", file=text_file)
        # How many products are in the subsetted categories?
        print('Products in OOS group: {}, products in Promo group: {}, '
              'Products in Normal group: {}'.format(test_df_oos['product_id'].nunique(),
                                                    test_df_promo['product_id'].nunique(),
                                                    test_df_normal['product_id'].nunique()), file=text_file)
        if len(test_df_oos) > 0:
            print(
                'OOS product types overall metrics [LGBM, WA] \n '
                'Huber: {:.5} {:.5} \n '
                'MSE: {:.5} {:.5} \n '
                'MAE: {:.5} {:.5} \n '
                'MAPE: {:.5} {:.5} \n '
                'wMAPE: {:.5} {:.5}'.format(
                    mean_huber(test_df_oos['actual'], test_df_oos['lgbm']), mean_huber(test_df_oos['actual'], test_df_oos['wa']),
                    mse(test_df_oos['actual'], test_df_oos['lgbm']), mse(test_df_oos['actual'], test_df_oos['wa']),
                    mae(test_df_oos['actual'], test_df_oos['lgbm']), mae(test_df_oos['actual'], test_df_oos['wa']),
                    mape(test_df_oos['actual'], test_df_oos['lgbm']), mape(test_df_oos['actual'], test_df_oos['wa']),
                    wmape(test_df_oos['actual'], test_df_oos['lgbm']), wmape(test_df_oos['actual'], test_df_oos['wa'])), file=text_file)

        # Promo types
        if len(test_df_promo) > 0:
            print(
                'Promo product types overall metrics [LGBM, WA] \n '
                'Huber: {:.5} {:.5} \n '
                'MSE: {:.5} {:.5} \n '
                'MAE: {:.5} {:.5} \n '
                'MAPE: {:.5} {:.5} \n '
                'wMAPE: {:.5} {:.5}'.format(
                    mean_huber(test_df_promo['actual'], test_df_promo['lgbm']), mean_huber(test_df_promo['actual'], test_df_promo['wa']),
                    mse(test_df_promo['actual'], test_df_promo['lgbm']), mse(test_df_promo['actual'], test_df_promo['wa']),
                    mae(test_df_promo['actual'], test_df_promo['lgbm']), mae(test_df_promo['actual'], test_df_promo['wa']),
                    mape(test_df_promo['actual'], test_df_promo['lgbm']), mape(test_df_promo['actual'], test_df_promo['wa']),
                    wmape(test_df_promo['actual'], test_df_promo['lgbm']), wmape(test_df_promo['actual'], test_df_promo['wa'])), file=text_file)

        # Normal types
        if len(test_df_normal) > 0:
            print(
                'Normal product types overall metrics [LGBM, WA] \n '
                'Huber: {:.5} {:.5} \n '
                'MSE: {:.5} {:.5} \n '
                'MAE: {:.5} {:.5} \n '
                'MAPE: {:.5} {:.5} \n '
                'wMAPE: {:.5} {:.5}'.format(
                    mean_huber(test_df_normal['actual'], test_df_normal['lgbm']), mean_huber(test_df_normal['actual'], test_df_normal['wa']),
                    mse(test_df_normal['actual'], test_df_normal['lgbm']), mse(test_df_normal['actual'], test_df_normal['wa']),
                    mae(test_df_normal['actual'], test_df_normal['lgbm']), mae(test_df_normal['actual'], test_df_normal['wa']),
                    mape(test_df_normal['actual'], test_df_normal['lgbm']), mape(test_df_normal['actual'], test_df_normal['wa']),
                    wmape(test_df_normal['actual'], test_df_normal['lgbm']), wmape(test_df_normal['actual'], test_df_normal['wa'])), file=text_file)
