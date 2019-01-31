############################################################################################
#                                                                                  [+] SETUP
############################################################################################

import numpy as np
import pandas as pd
import dask.dataframe as dd
import gc
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import lightgbm as lgb


# Load and initialize plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
# import plotly.io as pio


import warnings
warnings.simplefilter("ignore", RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

from custom_code.download_file_from_gcs import download_file_from_gcs

############################################################################################
#                                                                    [+] LOAD & PREPARE DATA
############################################################################################


def downcast_datatypes(df):
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int'])

    for cols in float_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='float')
    for cols in int_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='integer')

    return df


def transform_to_weekly_wa(df):
    """
    Select all rows in the test set that are not divisible by 7 (these are the first days of the weeks)
    Set all but the first days of the week to 0, then fill all NaN's with a forwardfill
    This propogates the first forecasted value of the week to all consecutive days in that week
    The end result is a fair 7-day ahead forecast for each week which is updated every 7 days
    """
    df['wa'][df.reset_index(drop=True).index % 7 != 0] = np.NaN
    df.fillna(method='ffill', inplace=True)
    return df

# Load data from GCS
results_df = pd.read_csv('./local_data/results_with_wa_40K_WA7_NEWCV.csv')
# results_df.drop_duplicates(subset=['product_id', 'date'], keep='first', inplace=True)

# Load WA forecasts
wa_df = pd.read_hdf('./local_data/wa.h5')

# Add OOF WA predictions to df
results_df = results_df.merge(wa_df, how='left', on=['date', 'product_id'])
results_df = results_df.dropna() # Drop all rows where there is no WA forecast (drops 2016/01)

# Load producttype info and add to forecast df (OOF predictions)
product_df = pd.read_hdf('./local_data/product.h5')
results_df = results_df.merge(product_df[['product_id', 'product_type_id']], on=['product_id'])

# Transform entire dataframe to weekly WA forecast
results_df = results_df[results_df['is_test'] == True] # Transform only the test folds for error computations <- for the metrics this is more accurate
results_df = results_df.groupby(['product_id', 'fold']).apply(transform_to_weekly_wa)
results_df = downcast_datatypes(results_df)


del wa_df
gc.collect()

############################################################################################
#                                                                       [+] VALIDATION TOOLS
############################################################################################


# Error metric functions
def mean_huber(y_true, y_pred, delta=1):
    huber = np.where(np.abs(y_true - y_pred) < delta, 0.5 * ((y_true - y_pred) ** 2),
                     delta * np.abs(y_true - y_pred) - 0.5 * (delta**2))
    return np.mean(huber)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = (np.abs((y_true - y_pred)) + 1) / (y_true + 1)
    return np.mean(ape)


def weighted_mape(y_true, y_pred):
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    return wmape


# Plotting functions
def plot_product_full(product_id):
    """
    Function to plot a comparison of the actuals, WA baseline model and LightGBM model
    for an individual product of choice. The plot displays the both the training data and OOF predictions.
    Corresponding Huber loss and MSE of the test folds is given in the title (LightGBM vs WA).
    """
    plot_df = results_df[results_df['product_id'] == product_id]
    plot_df['test_plot_ind'] = np.NaN  # Create a column to visualize test predictions throughout the series
    plot_df['test_plot_ind'][plot_df.is_test == True] = 0
    metric_df = plot_df[plot_df['is_test'] == True].copy()  # Only calculate metrics over the test predictions
    actual = go.Scattergl(y=plot_df['actual'], x=plot_df['date'],
                          name='Actuals', line=dict(color='#6FD05C'))
    lgbm_forecast = go.Scattergl(y=plot_df['lgbm'], x=plot_df['date'],
                                 name='LightGBM', line=dict(color='#00D7FC', dash='dash'))
    wa_forecast = go.Scattergl(y=plot_df['wa'], x=plot_df['date'],
                               name='WA Baseline', line=dict(color='#EC4785', dash='dash'))
    test_ind = go.Scattergl(y=plot_df['test_plot_ind'], x=plot_df['date'],
                            name='Test Indicator (OOF)', line=dict(color='#FC8036', width=4, dash='dot'))
    data = [actual, lgbm_forecast, wa_forecast, test_ind]
    title = '<b>LightGBM vs WA</b> <br> product_id: {} | Huber: {:.5} vs {:.5} | MSE: {:.5} vs {:.5}'.format(
        product_id, mean_huber(metric_df['actual'], metric_df['lgbm']), mean_huber(
            metric_df['actual'], metric_df['wa']), metrics.mean_squared_error(
            metric_df['actual'], metric_df['lgbm']), metrics.mean_squared_error(
            metric_df['actual'], metric_df['wa']))
    layout = go.Layout(title=title, yaxis=dict(title='Sales', zeroline=True),
                       font=dict(size=11, color='rgb(230,230,230)'),
                       autosize=False, width=1200, height=500, legend=dict(orientation='h', x=0, y=-0.1),
                       paper_bgcolor='#272C34',
                       plot_bgcolor='#272C34')
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def plot_error_boxplot(error='huber'):
    """
    """
    plot_df = results_df_grouped_metrics[['product_id', '{}_wa'.format(error), '{}_lgbm'.format(error)]]
    plot_df = plot_df[~plot_df.isin([np.nan, np.inf, -np.inf]).any(1)] # Drop NaNs and infs
    wa = go.Box(x=plot_df['{}_wa'.format(error)], name='WA Baseline', marker=dict(color='#EC4785'),
                boxmean='sd')
    lgbm = go.Box(x=plot_df['{}_lgbm'.format(error)], name='LightGBM', marker=dict(color='#00D7FC'),
                  boxmean='sd')
    data = [wa, lgbm]
    title = '<b>LightGBM vs WA</b> <br> Boxplot of {}'.format(error)
    layout = go.Layout(title=title, xaxis=dict(title='{} Error '.format(error), zeroline=True),
                       font=dict(size=11, color='rgb(230,230,230)'),
                       autosize=False, width=1200, height=300, legend=dict(orientation='h', x=0, y=-0.3),
                       paper_bgcolor='#272C34',
                       plot_bgcolor='#272C34',
                       hovermode='closest', showlegend=False)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def plot_error_histogram(error='wmape'):
    """
    """
    plot_df = results_df_grouped_metrics[['product_id', '{}_wa'.format(error), '{}_lgbm'.format(error)]]
    plot_df = plot_df[~plot_df.isin([np.nan, np.inf, -np.inf]).any(1)] # Drop NaNs and infs
    wa = go.Histogram(x=plot_df['{}_wa'.format(error)], name='WA Baseline', marker=dict(color='#EC4785'), opacity=0.7)
    lgbm = go.Histogram(x=plot_df['{}_lgbm'.format(error)], name='LightGBM', marker=dict(color='#00D7FC'), opacity=0.7)
    # wa_mean = go.Scatter(x=[results_df_grouped_metrics['{}_wa'.format(error)].mean()] * len(plot_df), y=list(range(5000)),
    #                      name='WA Mean Error', mode='lines', marker=dict(color='#EC4785'))
    # lgbm_mean = go.Scatter(x=[results_df_grouped_metrics['{}_lgbm'.format(error)].mean()] * len(plot_df), y=list(range(5000)),
    #                        name='LightGBM Mean Error', mode='lines', marker=dict(color='#00D7FC'))
    data = [wa, lgbm]
    title = '<b>LightGBM vs WA</b> <br> Histogram of {}'.format(error)
    layout = go.Layout(title=title, xaxis=dict(title='{} '.format(error), zeroline=True),
                       font=dict(size=11, color='rgb(230,230,230)'),
                       autosize=False, width=1200, height=400, legend=dict(x=0.8, y=1),
                       paper_bgcolor='#272C34',
                       plot_bgcolor='#272C34',
                       barmode='overlay')
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

############################################################################################
#                                                                        [+] COMPUTE METRICS
############################################################################################

# For the most accurate result first subset to weekly WA based on only the test dataset with the function above
# After this calculate the metrics with the weekly WA within the test folds

# Calculate metrics per product_id for all products (takes a couple mins to compute all metrics)
# results_df_grouped = results_df[results_df['is_test'] == True].dropna(how='any').groupby('product_id')
results_df_grouped = results_df.groupby('product_id')
results_df_grouped_metrics = pd.DataFrame()  # Create placeholder df
results_df_grouped_metrics['huber_lgbm'] = results_df_grouped.apply(
    lambda x: mean_huber(x['actual'], x['lgbm']))
results_df_grouped_metrics['huber_wa'] = results_df_grouped.apply(
    lambda x: mean_huber(x['actual'], x['wa']))
results_df_grouped_metrics['mse_lgbm'] = results_df_grouped.apply(
    lambda x: metrics.mean_squared_error(x['actual'], x['lgbm']))
results_df_grouped_metrics['mse_wa'] = results_df_grouped.apply(
    lambda x: metrics.mean_squared_error(x['actual'], x['wa']))
results_df_grouped_metrics['mae_lgbm'] = results_df_grouped.apply(
    lambda x: metrics.mean_absolute_error(x['actual'], x['lgbm']))
results_df_grouped_metrics['mae_wa'] = results_df_grouped.apply(
    lambda x: metrics.mean_absolute_error(x['actual'], x['wa']))
results_df_grouped_metrics['mape_lgbm'] = results_df_grouped.apply(
    lambda x: mape(x['actual'], x['lgbm']))
results_df_grouped_metrics['mape_wa'] = results_df_grouped.apply(
    lambda x: mape(x['actual'], x['wa']))
results_df_grouped_metrics['wmape_lgbm'] = results_df_grouped.apply(
    lambda x: weighted_mape(x['actual'], x['lgbm']))
results_df_grouped_metrics['wmape_wa'] = results_df_grouped.apply(
    lambda x: weighted_mape(x['actual'], x['wa']))

# Calculate differences between LightGBM and WA predictions
# Form is LightGBM_error - WA_error, such that that a negative error indicates LightGBM outperforms WA
results_df_grouped_metrics['huber_diff'] = results_df_grouped_metrics['huber_lgbm'] - \
    results_df_grouped_metrics['huber_wa']
results_df_grouped_metrics['mae_diff'] = results_df_grouped_metrics['mae_lgbm'] - \
    results_df_grouped_metrics['mae_wa']
results_df_grouped_metrics['mse_diff'] = results_df_grouped_metrics['mse_lgbm'] - \
    results_df_grouped_metrics['mse_wa']
results_df_grouped_metrics['mape_diff'] = results_df_grouped_metrics['mape_lgbm'] - \
    results_df_grouped_metrics['mape_wa']
results_df_grouped_metrics['wmape_diff'] = results_df_grouped_metrics['wmape_lgbm'] - \
    results_df_grouped_metrics['wmape_wa']

# Drop index and add back dropped product_types due to grouping
results_df_grouped_metrics.reset_index(drop=False, inplace=True)
results_df_grouped_metrics = results_df_grouped_metrics.merge(
    product_df[['product_id', 'product_type_id']], on=['product_id'])

# Mean errors
results_df_grouped_metrics['huber_lgbm'].mean(), results_df_grouped_metrics['huber_wa'].mean()
results_df_grouped_metrics['mse_lgbm'].mean(), results_df_grouped_metrics['mse_wa'].mean()
results_df_grouped_metrics['mae_lgbm'].mean(), results_df_grouped_metrics['mae_wa'].mean()
results_df_grouped_metrics['mape_lgbm'].mean(), results_df_grouped_metrics['mape_wa'].mean()
results_df_grouped_metrics_wmape = results_df_grouped_metrics[~results_df_grouped_metrics.isin(
    [np.nan, np.inf, -np.inf]).any(1)] # Drop NaNs and infs
results_df_grouped_metrics_wmape['wmape_lgbm'].mean(), results_df_grouped_metrics_wmape['wmape_wa'].mean()

# Median errors
results_df_grouped_metrics['huber_lgbm'].median(), results_df_grouped_metrics['huber_wa'].median()
results_df_grouped_metrics['mse_lgbm'].median(), results_df_grouped_metrics['mse_wa'].median()
results_df_grouped_metrics['mae_lgbm'].median(), results_df_grouped_metrics['mae_wa'].median()
results_df_grouped_metrics['mape_lgbm'].median(), results_df_grouped_metrics['mape_wa'].median()
results_df_grouped_metrics_wmape = results_df_grouped_metrics[~results_df_grouped_metrics.isin(
    [np.nan, np.inf, -np.inf]).any(1)] # Drop NaNs and infs
results_df_grouped_metrics_wmape['wmape_lgbm'].median(), results_df_grouped_metrics_wmape['wmape_wa'].median()

############################################################################################
#                                                                 [+] COMPUTE WEEKLY METRICS
############################################################################################

# For the most accurate result first subset to weekly WA based on only the test dataset with
# the transform_to_weekly_wa() function above
# After this, calculate the metrics with the weekly WA within the test folds

# Aggregate by week (a fold corresponds to a week for each product)
results_df_week = results_df.groupby(['product_id', 'fold']).sum().reset_index(drop=False)

# Calculate metrics per product_id for all products (takes a couple mins to compute all metrics)
results_df_grouped_week = results_df_week.groupby(['product_id'])
results_df_grouped_week_metrics = pd.DataFrame()  # Create placeholder df
results_df_grouped_week_metrics['huber_lgbm'] = results_df_grouped_week.apply(
    lambda x: mean_huber(x['actual'], x['lgbm']))
results_df_grouped_week_metrics['huber_wa'] = results_df_grouped_week.apply(
    lambda x: mean_huber(x['actual'], x['wa']))
results_df_grouped_week_metrics['mse_lgbm'] = results_df_grouped_week.apply(
    lambda x: metrics.mean_squared_error(x['actual'], x['lgbm']))
results_df_grouped_week_metrics['mse_wa'] = results_df_grouped_week.apply(
    lambda x: metrics.mean_squared_error(x['actual'], x['wa']))
results_df_grouped_week_metrics['mae_lgbm'] = results_df_grouped_week.apply(
    lambda x: metrics.mean_absolute_error(x['actual'], x['lgbm']))
results_df_grouped_week_metrics['mae_wa'] = results_df_grouped_week.apply(
    lambda x: metrics.mean_absolute_error(x['actual'], x['wa']))
results_df_grouped_week_metrics['mape_lgbm'] = results_df_grouped_week.apply(
    lambda x: mape(x['actual'], x['lgbm']))
results_df_grouped_week_metrics['mape_wa'] = results_df_grouped_week.apply(
    lambda x: mape(x['actual'], x['wa']))
results_df_grouped_week_metrics['wmape_lgbm'] = results_df_grouped_week.apply(
    lambda x: weighted_mape(x['actual'], x['lgbm']))
results_df_grouped_week_metrics['wmape_wa'] = results_df_grouped_week.apply(
    lambda x: weighted_mape(x['actual'], x['wa']))


# Drop index and add back dropped product_types due to grouping
results_df_grouped_week_metrics.reset_index(drop=False, inplace=True)
results_df_grouped_week_metrics = results_df_grouped_week_metrics.merge(
    product_df[['product_id', 'product_type_id']], on=['product_id'])


# Mean errors
results_df_grouped_week_metrics['huber_lgbm'].mean(), results_df_grouped_week_metrics['huber_wa'].mean()
results_df_grouped_week_metrics['mse_lgbm'].mean(), results_df_grouped_week_metrics['mse_wa'].mean()
results_df_grouped_week_metrics['mae_lgbm'].mean(), results_df_grouped_week_metrics['mae_wa'].mean()
results_df_grouped_week_metrics['mape_lgbm'].mean(), results_df_grouped_week_metrics['mape_wa'].mean()
results_df_grouped_week_metrics_wmape = results_df_grouped_week_metrics[~results_df_grouped_week_metrics.isin(
    [np.nan, np.inf, -np.inf]).any(1)] # Drop NaNs and infs
results_df_grouped_week_metrics_wmape['wmape_lgbm'].mean(), results_df_grouped_week_metrics_wmape['wmape_wa'].mean()

# Median errors
results_df_grouped_week_metrics['huber_lgbm'].median(), results_df_grouped_week_metrics['huber_wa'].median()
results_df_grouped_week_metrics['mse_lgbm'].median(), results_df_grouped_week_metrics['mse_wa'].median()
results_df_grouped_week_metrics['mae_lgbm'].median(), results_df_grouped_week_metrics['mae_wa'].median()
results_df_grouped_week_metrics['mape_lgbm'].median(), results_df_grouped_week_metrics['mape_wa'].median()
results_df_grouped_week_metrics_wmape = results_df_grouped_week_metrics[~results_df_grouped_week_metrics.isin(
    [np.nan, np.inf, -np.inf]).any(1)] # Drop NaNs and infs
results_df_grouped_week_metrics_wmape['wmape_lgbm'].median(), results_df_grouped_week_metrics_wmape['wmape_wa'].median()


############################################################################################
#                                                                       [+] SUMMARIZE ERRORS
############################################################################################
plot_error_boxplot('mae')
plot_error_boxplot('mse')
plot_error_boxplot('wmape')
plot_error_histogram('wmape')
plot_error_histogram('mape')

############################################################################################
#                                                                     [+] ZOOM INTO PRODUCTS
############################################################################################

# Find bad/good performing products in all products based on a metric of choice
results_df_grouped_metrics.sort_values('mse_lgbm', ascending=False).head(15)

############################################################################################
#                                                                      [+] VISUAL INSPECTION
############################################################################################

plot_product_full(154321)  # Veripart HDMI kabel Verguld 1,5 meter
plot_product_full(792487)  # Casio FX-CG50
plot_product_full(222837)  # Apple Lightning USB Ca ble
plot_product_full(723578)  # Trust Urban Primo Powerbank 4.400 mAh Zwart
plot_product_full(172298)  # Case Logic Sleeve 14" Zwart
plot_product_full(708189)  # Bestron Fan AFT760W


plot_product_full(773753)  # Samsung Galaxy Xcover 4
plot_product_full(594029)  # Beach chair
plot_product_full(469659)  # Fan
plot_product_full(222837)  # Apple Lightning USB cable
plot_product_full(170322)  # Logitech Wireless Mouse M23
plot_product_full(585051)  # Smart thermo
plot_product_full(772979)  # iPad cover
plot_product_full(766628)  # Vacuum cleaner
plot_product_full(342872)  # Airfryer
plot_product_full(749972)  # Apple Earpods Lightning
plot_product_full(566272)  # Fan
plot_product_full(117223)  # Bluetooth earphone
plot_product_full(233193)  # Apple charger
plot_product_full(479688)  # Gift card (50 euro)
plot_product_full(794555)  # AquaClean CA6903/10 Waterfilter
plot_product_full(430215)  # Some weird washing machine table
plot_product_full(659326)  # Fancy gaming keyboard
plot_product_full(672748)  # Wireless mouse
plot_product_full(567672)  # Iron
plot_product_full(785039)  # Seagate Game Drive PS4 2TB
plot_product_full(710628)  # TomTom
plot_product_full(585073)  # Washing machine
plot_product_full(773381)  # Apple iPhone SE 32GB Gold
plot_product_full(776276)  # Philips shaving
plot_product_full(741380)  # GoPro battery charger
