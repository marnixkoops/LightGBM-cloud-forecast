import tempfile
import subprocess

import gc
import pandas as pd

from custom_code.process_features import add_fold_aware_features
from custom_code.upload_file_to_gcs import upload_file_to_gcs
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.settings import NUM_FOLDS, RUNTAG, BUCKET, PROJECT, FEATURES_DIR


def create_fold_aware_features():
    for fold in list(range(0, NUM_FOLDS-1)):
        print('Generating fold aware features for fold {}'.format(fold))

        print('Reading feature matrix')
        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/train_x_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        train_x = pd.read_hdf(file_location, 'train_x')
        subprocess.call(['rm', '-f', file_location])

        file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/test_x_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        test_x = pd.read_hdf(file_location, 'test_x')
        subprocess.call(['rm', '-f', file_location])

        print('Creating fold aware features')
        train_x, test_x = add_fold_aware_features(train_x, test_x)

        print('Writing slice to GCS')
        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            train_x.to_hdf('{}.h5'.format(tf.name), 'train_x', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/train_x_complete_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
            # train_x.to_csv('{}.csv'.format(tf.name), index=False)
            # upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name), '{}/train_x_complete_{}_{}.csv'.format(FEATURES_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        # subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])
        del train_x

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            test_x.to_hdf('{}.h5'.format(tf.name), 'test_x', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/test_x_complete_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
            # test_x.to_csv('{}.csv'.format(tf.name), index=False)
            # upload_file_to_gcs(PROJECT, BUCKET, '{}.csv'.format(tf.name), '{}/test_x_complete_{}_{}.csv'.format(FEATURES_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        # subprocess.call(['rm', '-f', '{}.csv'.format(tf.name)])
        del test_x
        gc.collect()
