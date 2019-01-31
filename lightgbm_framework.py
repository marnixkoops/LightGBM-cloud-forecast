if __name__ == '__main__':
    from datetime import datetime
    import sys
    import subprocess
    import tempfile

    import gc
    from pyspark import SparkContext
    import pandas as pd

    from custom_code.upload_file_to_gcs import upload_file_to_gcs
    from custom_code.download_file_from_gcs import download_file_from_gcs
    from custom_code.process_features import process_features, downcast_datatypes
    from custom_code.train_model import train_model
    from custom_code.settings import RUNTAG, PROJECT, BUCKET, DATA_DIR

    sc = SparkContext.getOrCreate()
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    LOGGER.info('LightGBM calculation started')

    _, kwargs_string = sys.argv
    runner_kwargs = eval(kwargs_string)

    project = runner_kwargs['project']

    LOGGER.info('Starting process to generate features')
    try:
        LOGGER.info('Attempting to download existing features file in GCS for current runtag: {}'.format(RUNTAG))
        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/features.h5'.format(DATA_DIR))
        features_df = pd.read_hdf(file_location, 'features_df')
        subprocess.call(['rm', '-f', file_location])
        LOGGER.info('Downcasting datatypes for entire feature matrix')
        features_df = downcast_datatypes(features_df)
    except IOError:
        LOGGER.info('Failed to find existing features file for current runtag {}, '
                    'so features will be generated from scratch'.format(RUNTAG))
        LOGGER.info('Downloading data (actuals) hdf file')
        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/actual.h5'.format(DATA_DIR))
        LOGGER.info('Reading data (actuals) in pandas dataframe')
        data_df = pd.read_hdf(file_location, 'data_df')
        LOGGER.info('Starting feature generation')
        features_df = process_features(data_df, LOGGER)
        subprocess.call(['rm', '-f', file_location])
        LOGGER.info('Writing features dataframe as local hdf file')
        features_df.to_hdf('./features.h5', 'features_df', index=False)
        LOGGER.info('Uploading features dataframe to GCS')
        upload_file_to_gcs(PROJECT, BUCKET, './features.h5', '{}/features.h5'.format(DATA_DIR))
        del data_df
        gc.collect()

    LOGGER.info('Starting process to train and predict with LightGBM')
    train_model(features_df, LOGGER)
    gc.collect()
