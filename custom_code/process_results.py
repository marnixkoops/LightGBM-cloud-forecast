import subprocess

from sklearn import metrics
import pandas as pd
import dask.dataframe as dd

from custom_code.metrics import mean_huber, mape, wmape
from custom_code.plotting import plot_importances, plot_shap_importances
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.settings import RUNTAG, COMPUTE_SHAP, PROJECT, BUCKET, RESULTS_DIR, DATA_DIR


def write_metrics(test_df, filepath, features_names, params):
    print('Calculating metrics ...')

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

    print('Overall Huber LGBM: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
            overall_huber_lgbm, overall_mse_lgbm, overall_mae_lgbm, overall_mape_lgbm, overall_wmape_lgbm))

    print('Overall Huber WA: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
            overall_huber_wa, overall_mse_wa, overall_mae_wa, overall_mape_wa, overall_wmape_wa))

    model_id = RUNTAG + '_Huber_{:.5}'.format(overall_huber_lgbm)

    # Write model results to log
    print('Writing model info and results to log ...')
    with open(filepath, "a+") as text_file:
        print(
            "[----------------------------------------------------------------------------------------------------------]",
            file=text_file)
        print("Model ID: {}".format(model_id), file=text_file)
        print("Test Range: from {} to {}".format(test_df['date'].min(), test_df['date'].max()), file=text_file)

        print('LGBM', file=text_file)
        print('Overall Huber: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
                overall_huber_lgbm, overall_mse_lgbm, overall_mae_lgbm, overall_mape_lgbm, overall_wmape_lgbm), file=text_file)

        print('WA', file=text_file)
        print('Overall Huber: {:.5}, Overall MSE: {:.5}, Overall MAE: {:.5}, Overall MAPE: {:.5}, Overall Weighted MAPE: {:.5}'.format(
                overall_huber_wa, overall_mse_wa, overall_mae_wa, overall_mape_wa, overall_wmape_wa), file=text_file)
        print("Parameters: {}".format(params), file=text_file)
        print("Features: {}".format(features_names), file=text_file)

        n_folds = max(test_df['fold'])
        for fold in range(n_folds + 1):
            fold_test_df = test_df[test_df.fold == fold]
            print('Fold {} from {} to {}'.format(fold, fold_test_df['date'].min(), fold_test_df['date'].max()),
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


def process_results(results_df, feature_importance_df, params):
    print('Processing results ...')
    # Join with baseline model
    file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/wa_{}.h5'.format(DATA_DIR, 'ALL_PRODUCTS_LESS_FEATURES'))
    wa_df = pd.read_hdf(file_location, 'wa_forecast_df')
    results_df = pd.merge(results_df, wa_df, how='left', on=['date', 'product_id'])
    subprocess.call(['rm', '-f', file_location])
    del wa_df

    features_names = list(feature_importance_df[feature_importance_df.fold == 0]['feature'])

    # Create a DF that only covers the test periods and when products are on stock
    # We may not have WA for all dates within test, so drop those that don't
    test_df = results_df[results_df.is_test & results_df.on_stock].dropna()
    write_metrics(test_df, 'lgbm_log.txt', features_names, params)

    # Plotting
    print('Saving feature importance plots to GCS ...')
    # Only read SHAP if they were computed
    if COMPUTE_SHAP is True:
        # shap_df = dd.read_csv('gs://{}/{}/shap_*_{}.csv'.format(BUCKET, RESULTS_DIR, RUNTAG))
        # shap_df = shap_df.compute()
        shap_df = dd.read_csv('./shap_*_{}.csv'.format(RUNTAG))
        shap_df = shap_df.compute()

    for fold in range(max(feature_importance_df['fold'] + 1)):
        fold_feature_importance_df = feature_importance_df[feature_importance_df.fold == fold].drop('fold', axis=1)
        plot_importances(fold_feature_importance_df, fold, type='split')
        plot_importances(fold_feature_importance_df, fold, type='gain')

        # Only plot SHAP if they were computed
        if COMPUTE_SHAP:
            fold_shap_df = shap_df[shap_df.fold == fold].drop(['fold', 'date'], axis=1)
            plot_shap_importances(fold_shap_df.values, fold_shap_df.columns, fold)
            del fold_shap_df

        del fold_feature_importance_df

    return results_df
