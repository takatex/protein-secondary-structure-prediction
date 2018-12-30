# -*- coding: utf-8 -*-
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from make_dataset import download_dataset, make_datasets


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

    def __init__(self, data_dir, batch_size_train, batch_size_test):
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'train.npz')
        self.test_path = os.path.join(data_dir, 'test.npz')
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_dataset(self):
        if not(os.path.isfile(self.train_path) and os.path.isfile(self.test_path)):
            download_dataset()
            make_datasets()

        # train dataset
        train_data = np.load(self.train_path)
        X_train, y_train, seq_len_train = train_data['X'], train_data['y'], train_data['seq_len']

        # test dataset
        test_data = np.load(self.test_path)
        X_test, y_test, seq_len_test = test_data['X'], test_data['y'], test_data['seq_len']

        return X_train, y_train, seq_len_train, X_test, y_test, seq_len_test

    def __call__(self):
        X_train, y_train, seq_len_train, X_test, y_test, seq_len_test = \
            self.load_dataset()

        D_train = MyDataset(X_train, y_train, seq_len_train)
        train_loader = DataLoader(D_train, batch_size=self.batch_size_train, shuffle=True)

        D_test = MyDataset(X_test, y_test, seq_len_test)
        test_loader = DataLoader(D_test, batch_size=self.batch_size_test, shuffle=False)

        return train_loader, test_loader
