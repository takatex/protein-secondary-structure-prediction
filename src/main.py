# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.pardir)
import numpy as np
import pickle
import time
import argparse
import collections
from utils import *
# from visualizer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from data import LoadDataset
from model import Net
from sklearn.model_selection import KFold

# params
# ----------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300,
                        help='The number of epochs to run (default: 300)')
parser.add_argument('--batch_size', type=int, default=64,
                        help='The number of batch (default: 64)')
parser.add_argument('--hidden_size', type=int, default=128,
                        help='The number of features in the hidden state h (default: 256)')
parser.add_argument('--num_layers', type=int, default=3,
                        help='The number of layers (default: 3)')
parser.add_argument('--result_path', type=str, default='./result',
                        help='Result path (default: ./result)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

args = parser.parse_args()


def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    len_ = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        show_progress(epoch, batch_idx, len_, loss.item())

    return loss.item()


def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    acc = 0
    len_ = len(test_loader)
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).cpu().numpy() # sum up batch loss
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            acc += np.abs(t - o)

    test_loss /= len_
    acc /= len_

    return test_loss, acc



def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    result_path = os.path.join(args.result_path, args.model)
    os.makedirs(result_path, exist_ok=True)

    # laod dataset and set k-fold cross validation
    D = LoadDataset(batch_size_train=1, batch_size_test=1)
    idxs = np.arange(D.__len__())
    kf = KFold(n_splits=args.k)

    for k, idx in enumerate(kf.split(idxs)):
        train_loader, test_loader = D[idx]

        # model, loss_function, optimizer
        model = Net().to(device).double()
        loss_function = load_loss_function()
        optimizer = torch.optim.Adam(model.parameters())

        # train and test
        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, device, train_loader, optimizer, loss_function, epoch)
            test_loss, count = test(model, device, test_loader, loss_function)
            history.append([train_loss, test_loss, hard_acc, soft_acc])
            show_progress(epoch, train_loss, test_loss, hard_acc, soft_acc)

        # save train history and model
        np.save(os.path.join(result_path, f'history_{args.model}.npy'), np.array(history))
        torch.save(model.state_dict(), os.path.join(result_path, f'{args.model}.pth'))

if __name__ == '__main__':
    main()
