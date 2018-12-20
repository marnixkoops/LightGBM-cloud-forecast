if __name__ == '__main__':
    import sys
    import subprocess
    import tempfile

    import gc
    from pyspark import SparkContext
    import pandas as pd

    from custom_code.upload_file_to_gcs import upload_file_to_gcs
    from custom_code.download_file_from_gcs import download_file_from_gcs
    from custom_code.process_features import process_features
    from custom_code.create_folds_and_slice_features import create_folds_and_slice_features
    from custom_code.create_fold_aware_features import create_fold_aware_features
    from custom_code.train_model_per_fold import train_model_per_fold
    from custom_code.predict_and_save_results_per_fold import predict_and_save_results_per_fold
    from custom_code.settings import RUNTAG, PROJECT, BUCKET, DATA_DIR

    sc = SparkContext.getOrCreate()
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    LOGGER.info('LightGBM calculation started')

    _, kwargs_string = sys.argv
    runner_kwargs = eval(kwargs_string)

    project = runner_kwargs['project']
    features = runner_kwargs['features']
    folds_and_slices = runner_kwargs['folds_and_slices']
    fold_aware_features = runner_kwargs['fold_aware_features']
    train_per_fold = runner_kwargs['train_per_fold']
    predict_per_fold = runner_kwargs['predict_per_fold']

    if features is True:
        LOGGER.info('Starting process to generate features')
        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/actual_{}.h5'.format(DATA_DIR, RUNTAG))
        data_df = pd.read_hdf(file_location, 'data_df')
        features_df = process_features(data_df)
        subprocess.call(['rm', '-f', file_location])
        del data_df

        LOGGER.info('Writing features to GCS')
        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            features_df.to_hdf('{}.h5'.format(tf.name), 'features_df', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/features_{}.h5'.format(DATA_DIR, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        del features_df
        gc.collect()

    if folds_and_slices is True:
        LOGGER.info('Creating folds and slicing features')
        create_folds_and_slice_features()
        gc.collect()

    if fold_aware_features is True:
        LOGGER.info('Creating fold aware features per slice')
        create_fold_aware_features()
        gc.collect()

    if train_per_fold is True:
        LOGGER.info('Training lgb model')
        train_model_per_fold()
        gc.collect()

    if predict_per_fold is True:
        LOGGER.info('Predicting with lgb model')
        predict_and_save_results_per_fold()
        gc.collect()
