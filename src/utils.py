# -*- coding: utf-8 -*-

import os
import time
import platform
import numpy as np
import gzip
import collections

import torch
from torch import nn


def loss_func(out, target, seq_len):
    loss = 0
    for o, t, l in zip(out, target, seq_len):
        loss += nn.CrossEntropyLoss()(o[:l], t[:l])
    return loss

# class LossFunc(object):
#
#     def __init__(self):
#         self.loss = nn.CrossEntropyLoss()
#
#     def __call__(self, out, target, seq_len):
#         """
#         out.shape : (batch_size, class_num, seq_len)
#         target.shape : (batch_size, seq_len)
#         """
#         out = torch.clamp(out, 1e-15, 1 - 1e-15)
#         return torch.tensor([self.loss(o[:l], t[:l])
#                              for o, t, l in zip(out, target, seq_len)],
#                             requires_grad=True).sum()


def accuracy(out, target, seq_len):
    """
    out.shape : (batch_size, seq_len, class_num)
    target.shape : (class_num, seq_len)
    seq_len.shape : (batch_size)
    """
    out = out.cpu().data.numpy()
    target = target.cpu().data.numpy()
    seq_len = seq_len.cpu().data.numpy()

    out = out.argmax(axis=2)
    return np.array([np.equal(o[:l], t[:l]).sum()/l
                     for o, t, l in zip(out, target, seq_len)]).mean()


def amino_count(t):
    c = collections.Counter(t)
    keys, values = c.keys(), c.values()
    return list(keys), list(values)


def acid_accuracy(out, target, seq_len):
    out = out.cpu().data.numpy()
    target = target.cpu().data.numpy()
    seq_len = seq_len.cpu().data.numpy()

    out = out.argmax(axis=2)

    count_1 = np.zeros(8)
    count_2 = np.zeros(8)
    for o, t, l in zip(out, target, seq_len):
        o, t = o[:l], t[:l]

        # org
        keys, values = amino_count(t)
        count_1[keys] += values

        # pred
        keys, values = amino_count(t[np.equal(o, t)])
        count_2[keys] += values

    return np.divide(count_2, count_1, out=np.zeros(8), where=count_1!=0)


def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


def timestamp():
    return time.strftime("%Y%m%d%H%M", time.localtime())

# def show_progress(k, e, b, b_total, loss):
#     print(f'\r{e:3d} : [{b:3d} / {b_total:3d}] loss : {loss:.2f}', end='')

def show_progress(k, k_total, e, e_total):
    print(f'k:({k:2d}/{k_total:2d}), e:({e:3d}/{e_total:3d})')

def save_history(k, history, save_dir):
    save_path = os.path.join(save_dir, f'history_{k}.npy')
    np.save(save_path, history)

def save_model(k, model, save_dir):
    save_path = os.path.join(save_dir, f'model_{k}.pth')
    torch.save(model.state_dict(), save_path)
