# -*- coding: utf-8 -*-
import os
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from download_dataset import download_dataset

DATASET_PATH = '../data/cb513/cb513.npz'

class MyDataset(Dataset):

    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y.astype(int)
        self.seq_len = seq_len.astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        seq_len = self.seq_len[idx]
        return x, y, seq_len

class LoadDataset(object):

    def __init__(self, batch_size_train, batch_size_test):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.X, self.y, self.seq_len = self.load_dataset()

    def load_dataset(self):
        if not os.path.isfile(DATASET_PATH):
            download_dataset()
        else:
            pass

        loaded = np.load(DATASET_PATH)
        return loaded['X'], loaded['y'], loaded['seq_len']

    def __len__(self):
        return len(self.X)

    def __call__(self, idx):
        train_idx, test_idx = idx
        X_train, y_train, seq_len_train, X_test, y_test, seq_len_test = \
            self.X[train_idx], self.y[train_idx], self.seq_len[train_idx], \
            self.X[test_idx], self.y[test_idx], self.seq_len[test_idx]

        D_train = MyDataset(X_train, y_train, seq_len_train)
        train_loader = DataLoader(D_train, batch_size=self.batch_size_train, shuffle=True)

        D_test = MyDataset(X_test, y_test, seq_len_test)
        test_loader = DataLoader(D_test, batch_size=self.batch_size_test, shuffle=False)

        return train_loader, test_loader
