# -*- coding: utf-8 -*-

import os
import time
import json
import pickle
import numpy as np
import gzip
import collections

import torch
from torch import nn


def save_text(data, save_path):
    with open(save_path, mode='w') as f:
        f.write('\n'.join(data))


def save_picke(data, save_path):
    with open(save_path, mode="wb") as f:
        pickle.dump(data, f)


def args2json(data, path, print_args=True):
    data = vars(data)
    if print_args:
        print(f'\n+ ---------------------------')
        for k, v in data.items():
            print(f'  {k.upper()} : {v}')
        print(f'+ ---------------------------\n')

    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(data, f)



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

def show_progress(e, e_total, train_loss, test_loss, acc):
    print(f'[{e:3d}/{e_total:3d}] train_loss:{train_loss:.2f}, '\
          f'test_loss:{test_loss:.2f}, acc:{acc:.3f}')


def save_history(history, save_dir):
    save_path = os.path.join(save_dir, 'history.npy')
    np.save(save_path, history)


def save_model(model, save_dir):
    save_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), save_path)
