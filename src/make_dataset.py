# -*- coding: utf-8 -*-
# Original Code : https://github.com/alrojo/CB513/blob/master/data.py

import os
import numpy as np
import subprocess
from utils import load_gz


DATA_DIR = '../data'
TRAIN_PATH = os.path.join(DATA_DIR, 'cullpdb+profile_6133_filtered.npy.gz')
TEST_PATH = os.path.join(DATA_DIR, 'cb513+profile_split1.npy.gz')
TRAIN_DATASET_PATH = os.path.join(DATA_DIR, 'train.npz')
TEST_DATASET_PATH = os.path.join(DATA_DIR, 'test.npz')
TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"


def download_dataset():
    print('[Info] Downloading CB513 dataset ...')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
    os.system(f'wget -O {TEST_PATH} {TEST_URL}')


def make_datasets():
    print('[Info] Making datasets ...')

    # train dataset
    X_train, y_train, seq_len_train = make_dataset(TRAIN_PATH)
    np.savez_compressed(TRAIN_DATASET_PATH, X=X_train, y=y_train, seq_len=seq_len_train)
    print(f'[Info] Saved train dataset in {TRAIN_DATASET_PATH}')

    # test dataset
    X_test, y_test, seq_len_test = make_dataset(TEST_PATH)
    np.savez_compressed(TEST_DATASET_PATH, X=X_test, y=y_test, seq_len=seq_len_test)
    print(f'[Info] Saved test dataset in {TEST_DATASET_PATH}')


def make_dataset(path):
    data = load_gz(path)
    data = data.reshape(-1, 700, 57)

    idx = np.append(np.arange(21), np.arange(35, 56))
    X = data[:, :, idx]
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(8)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype('float32')

    return X, y, seq_len
