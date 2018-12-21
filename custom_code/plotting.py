import matplotlib as mpl
mpl.use('TkAgg')
import tempfile
import matplotlib.pyplot as plt
import shap
import seaborn as sns
# Load and initialize plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from custom_code.settings import RUNTAG, PROJECT, BUCKET, PLOTS_DIR
from custom_code.upload_file_to_gcs import upload_file_to_gcs

init_notebook_mode(connected=True)


# Split / gain feature importances
def plot_importances(feature_importance_df, type='gain'):
    file_location = tempfile.NamedTemporaryFile(delete=False).name
    cols = feature_importance_df[['feature', '{}'.format(type)]].groupby('feature').mean().index
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)].sort_values(
        by='{}'.format(type), ascending=False)
    plt.figure(figsize=(9, 12))
    sns.barplot(
        y='feature',
        x=type,
        data=best_features.sort_values(by=type, ascending=False)
    )
    plt.tight_layout()
    plt.title('mean {} importance (Over folds + Std dev)'.format(type))
    plt.savefig('{}.png'.format(file_location))
    plt.clf()

    upload_file_to_gcs(PROJECT, BUCKET, '{}.png'.format(file_location), '{}/imp_{}_overall_{}.png'.format(PLOTS_DIR, type, RUNTAG))


# Simple SHAP feature importances
def plot_shap_importances(shap_values, features_names, fold):
    file_location = tempfile.NamedTemporaryFile(delete=False).name
    shap.summary_plot(shap_values, features_names, plot_type='bar', show=False, max_display=len(features_names), auto_size_plot=True)
    plt.title('mean SHAP importance')
    plt.savefig('{}.png'.format(file_location))
    plt.clf()

    upload_file_to_gcs(PROJECT, BUCKET, '{}.png'.format(file_location), '{}/imp_shap_{}_{}.png'.format(PLOTS_DIR, fold, RUNTAG))


def plot_product(product_id, results_df, metrics_df, fold, results_subset_df=None, results_productid_df=None):
    """
    Function to plot a comparison of the actuals, WA baseline model and LightGBM predictions
    for an individual product of choice. The plot displays the both the training data and OOF predictions.
    Also, a rough indication of the training / validation folds is shown.
    Corresponding Huber loss and MSE is given in the title (LightGBM vs WA).
    """
    plot_df = results_df[(results_df.product_id == product_id) & (results_df.fold == fold)].fillna(-1)
    metrics_product_df = metrics_df[metrics_df['product_id'] == product_id]
    # LightGBM has missing in-sample predictions, fill with -1 for now
    actual = go.Scattergl(y=plot_df['actual'], x=plot_df['date'], name='Actuals', line=dict(color='darkorange'))
    lgbm_train = go.Scattergl(y=plot_df[plot_df.is_test==False]['lgbm'], x=plot_df[plot_df.is_test==False]['date'], name='LightGBM Train', line=dict(color='blue'))
    lgbm_test = go.Scattergl(y=plot_df[plot_df.is_test]['lgbm'], x=plot_df[plot_df.is_test]['date'], name='LightGBM Test', line=dict(color='blue', dash='dash'))
    if results_subset_df is not None:
        plot_subset_df = results_subset_df[(results_subset_df.product_id == product_id) & (results_subset_df.fold == fold)].fillna(-1)
        lgbm_train_subset = go.Scattergl(y=plot_subset_df[plot_subset_df.is_test == False]['lgbm'], x=plot_subset_df[plot_subset_df.is_test == False]['date'], name='LightGBM Subset Train', line=dict(color='magenta'))
        lgbm_test_subset = go.Scattergl(y=plot_subset_df[plot_subset_df.is_test]['lgbm'], x=plot_subset_df[plot_subset_df.is_test]['date'], name='LightGBM Subset Test', line=dict(color='magenta', dash='dash'))
    if results_productid_df is not None:
        plot_productid_df = results_productid_df[(results_productid_df.product_id == product_id) & (results_productid_df.fold == fold)].fillna(-1)
        lgbm_train_productid = go.Scattergl(y=plot_productid_df[plot_productid_df.is_test == False]['lgbm'], x=plot_productid_df[plot_productid_df.is_test == False]['date'], name='LightGBM Productid Train', line=dict(color='black'))
        lgbm_test_productid = go.Scattergl(y=plot_productid_df[plot_productid_df.is_test]['lgbm'], x=plot_productid_df[plot_productid_df.is_test]['date'], name='LightGBM Productid Test', line=dict(color='black', dash='dash'))
    wa = go.Scattergl(y=plot_df['wa'], x=plot_df['date'], name='Weighted Average', line=dict(color='green', dash='dash'))
    oos_df = plot_df[plot_df.on_stock == False][['date', 'on_stock']]
    oos = go.Scattergl(y=oos_df['on_stock'].astype('int'), x=oos_df['date'], name='Out Of Stock', mode='markers', marker=dict(size=4, color='red'))
    if results_subset_df is None and results_productid_df is None:
        data = [actual, lgbm_train, lgbm_test, wa, oos]
    if results_subset_df is not None and results_productid_df is not None:
        data = [actual, lgbm_train, lgbm_test, lgbm_train_subset, lgbm_test_subset, lgbm_train_productid, lgbm_test_productid, wa, oos]
    if results_subset_df is not None and results_productid_df is None:
        data = [actual, lgbm_train, lgbm_test, lgbm_train_subset, lgbm_test_subset, wa, oos]
    if results_subset_df is None and results_productid_df is not None:
        data = [actual, lgbm_train, lgbm_test, lgbm_train_productid, lgbm_test_productid, wa, oos]
    title = '<b>LightGBM vs WA Fold {}</b> <br> product_id: {} | Huber: {:.5} vs {:.5} | MSE: {:.5} vs {:.5}'.format(
        fold,
        product_id,
        float(metrics_product_df['huber_lgbm']),
        float(metrics_product_df['huber_wa']),
        float(metrics_product_df['mse_lgbm']),
        float(metrics_product_df['mse_wa'])
    )

    layout = go.Layout(title=title, yaxis=dict(title='Sales'))
    iplot(go.Figure(data=data, layout=layout))
