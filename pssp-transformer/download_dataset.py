# -*- coding: utf-8 -*-
# Original Code : https://github.com/alrojo/CB513/blob/master/data.py

import os
import numpy as np
import subprocess
from utils import load_gz, save_text, save_picke


TRAIN_PATH = '../pssp-data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '../pssp-data/cb513+profile_split1.npy.gz'

TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"

AA_PATH = lambda key : f'../pssp-data/aa_{key}.txt'
SP_PATH = lambda key : f'../pssp-data/sp_{key}.pkl'
PSS_PATH = lambda key : f'../pssp-data/pss_{key}.txt'


def download_dataset():
    print('[Info] Downloading CB513 dataset ...')
    if not (os.path.isfile(TRAIN_PATH) and os.path.isfile(TEST_PATH)):
        os.makedirs('../pssp-data', exist_ok=True)
        os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
        os.system(f'wget -O {TEST_PATH} {TEST_URL}')


def make_datasets():
    print('[Info] Making datasets ...')

    # train dataset
    X_train, y_train, seq_len_train = make_dataset(TRAIN_PATH)
    make_dataset_for_transformer(X_train, y_train, seq_len_train, 'train')

    # test dataset
    X_test, y_test, seq_len_test = make_dataset(TEST_PATH)
    make_dataset_for_transformer(X_test, y_test, seq_len_test, 'test')


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
    seq_len = seq_len.astype(int)

    return X, y, seq_len


def make_dataset_for_transformer(X, y, seq_len, key):
    X_amino = X[:, :21, :]
    X_profile = X[:, 21:, :]

    amino_acid_array = get_amino_acid_array(X_amino, seq_len)
    save_path = AA_PATH(key)
    save_text(amino_acid_array, save_path)
    print(f'[Info] Saved amino_acid_array for {key} in {save_path}')

    seq_profile = get_seq_profile(X_profile, seq_len)
    save_path = SP_PATH(key)
    save_picke(seq_profile, save_path)
    print(f'[Info] Saved seq_profile for {key} in {save_path}')

    pss_array = get_pss_array(y, seq_len)
    save_path = PSS_PATH(key)
    save_text(pss_array, save_path)
    print(f'[Info] Saved pss_array for {key} in {save_path}')


def get_amino_acid_array(X_amino, seq_len):
    amino_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',
                  'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    amino_acid_array = []
    for X, l in zip(X_amino, seq_len):
        acid = {}
        for i, aa in enumerate(amino_acid):
            keys = np.where(X[i] == 1)[0]
            values = [aa] * len(keys)
            acid.update(zip(keys, values))
        aa_str = ' '.join([acid[i] for i in range(l)])

        amino_acid_array.append(aa_str)
    return amino_acid_array


def get_pss_array(label, seq_len):
    pss_icon = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    pss_array = []
    for target, l in zip(label, seq_len):
        pss = np.array(['Nofill'] * l)
        target = target[:l]
        for i, p in enumerate(pss_icon):
            idx = np.where(target == i)[0]
            pss[idx] = p

        pss_str = ' '.join([pss[i] for i in range(l)])
        pss_array.append(pss_str)

    return pss_array


def get_seq_profile(X_profile, seq_len):
    seq_profile = []
    for sp, l in zip(X_profile, seq_len):
        seq_profile.append(sp[:, :l])
    return seq_profile
