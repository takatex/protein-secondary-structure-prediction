# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.pardir)
import numpy as np
import argparse
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim

from data import LoadDataset
from model import Net
from utils import *

# params
# ----------
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-e', '--epochs', type=int, default=1000,
                    help='The number of epochs to run (default: 1000)')
parser.add_argument('-b', '--batch_size_train', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--save_dir', type=str, default='../data/result',
                    help='Result path (default: ../data/result)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()


def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0
    len_ = len(train_loader)
    for batch_idx, (data, target, seq_len) in enumerate(train_loader):
        data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target, seq_len)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(f'\rloss={loss.item()/len(data):.5f}', end=', ')

    train_loss /= len_
    print(f'loss={train_loss:.5f}', end=', ')

    return train_loss


def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    acc = 0
    len_ = len(test_loader)
    with torch.no_grad():
        for i, (data, target, seq_len) in enumerate(test_loader):
            data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
            out = model(data)
            test_loss += loss_function(out, target, seq_len).cpu().data.numpy() # sum up batch loss
            acc += accuracy(out, target, seq_len)

    test_loss /= len_
    acc /= len_
    print(f'acc={acc:.2f}', end=', ')

    return test_loss, acc


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # make directory to save train history and model
    save_dir = f'{args.save_dir}_{timestamp()}'
    os.makedirs(save_dir, exist_ok=True)

    # laod dataset and set k-fold cross validation
    D = LoadDataset(args.batch_size_train, args.batch_size_test)

    train_loader, test_loader = D(idx)

    # model, loss_function, optimizer
    model = Net().to(device)
    loss_function = loss_func
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

    # train and test
    history = []
    for e in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, loss_function)
        test_loss, acc = test(model, device, test_loader, loss_function)
        history.append([train_loss, test_loss, acc])
        show_progress(k+1, args.k_fold, e, args.epochs)

    # save train history and model
    save_history(k, history, save_dir)
    save_model(k, model, save_dir)

if __name__ == '__main__':
    main()
