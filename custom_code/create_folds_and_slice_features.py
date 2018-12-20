import tempfile
import subprocess

import gc
import pandas as pd

from custom_code import timefold
from custom_code.upload_file_to_gcs import upload_file_to_gcs
from custom_code.download_file_from_gcs import download_file_from_gcs
from custom_code.settings import RUNTAG, PROJECT, BUCKET, DATA_DIR, FEATURES_DIR


def create_folds_and_slice_features():
    print('Loading full feature matrix')
    file_location = download_file_from_gcs(PROJECT, BUCKET, '{}/features_{}.h5'.format(DATA_DIR, RUNTAG))
    features_df = pd.read_hdf(file_location, 'features_df')
    target = features_df['actual'].copy()
    subprocess.call(['rm', '-f', file_location])

    min_train_size = int(0.75 * features_df.shape[0])  # 2+ years training
    min_test_size = int(0.03 * features_df.shape[0])  # 2+ months of testing
    step_size = int(0.03 * features_df.shape[0])  # 2+ month step size between folds
    timefolds = timefold.timefold(method='step', min_train_size=min_train_size, min_test_size=min_test_size, step_size=step_size)

    folds = {}
    for fold, (train_idx, test_idx) in enumerate(timefolds.split(features_df)):
        print('Generating fold {}'.format(fold))
        folds[str(fold)] = (list(map(int, train_idx)), list(map(int, test_idx)))

    for fold, (train_idx, test_idx) in folds.items():
        print('Generating feature slice for fold {}'.format(fold))
        print('Train idx: {} to {}, test idx: {} to {}'.format(train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))
        train_x, train_y = features_df.iloc[train_idx], target.iloc[train_idx]
        test_x, test_y = features_df.iloc[test_idx], target.iloc[test_idx]

        print('Writing slices to GCS')
        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            train_x.to_hdf('{}.h5'.format(tf.name), 'train_x', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/train_x_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        del train_x

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            train_y.to_hdf('{}.h5'.format(tf.name), 'train_y', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/train_y_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        del train_y

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            test_x.to_hdf('{}.h5'.format(tf.name), 'test_x', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/test_x_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        del test_x

        with open(tempfile.NamedTemporaryFile().name, 'w') as tf:
            test_y.to_hdf('{}.h5'.format(tf.name), 'test_y', index=False)
            upload_file_to_gcs(PROJECT, BUCKET, '{}.h5'.format(tf.name), '{}/test_y_{}_{}.h5'.format(FEATURES_DIR, fold, RUNTAG))
        subprocess.call(['rm', '-f', tf.name])
        subprocess.call(['rm', '-f', '{}.h5'.format(tf.name)])
        del test_y
        gc.collect()
