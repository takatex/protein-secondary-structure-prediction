import time
import platform
import numpy as np 
import gzip

import torch
from torch import nn

class LossFunc(object):

    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, out, target, seq_len):
        """
        out.shape : (batch_size, class_num, seq_len)
        target.shape : (batch_size, seq_len)
        """
        out = torch.clamp(out, 1e-15, 1 - 1e-15)
        return torch.Tensor([self.loss(o[:l], t[:l])
                             for o, t, l in zip(out, target, seq_len)]).sum()


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


def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)

######################
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def log_losses(y, t, eps=1e-15):
    if t.ndim == 1:
        t = one_hot(t)

    y = np.clip(y, eps, 1 - eps)
    losses = -np.sum(t * np.log(y), axis=1)
    return losses

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.mean(losses)

def proteins_acc(out, label, mask):
    out = np.argmax(out, axis=2)
    return np.sum(((out == label).flatten()*mask.flatten())).astype('float32') / np.sum(mask).astype('float32')


def accuracy(y, t):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
        
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

def entropy(x):
    h = -x * np.log(x)
    h[np.invert(np.isfinite(h))] = 0
    return h.sum(1)


def accuracy_topn(y, t, n=5):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    
    predictions = np.argsort(y, axis=1)[:, -n:]    
    
    accs = np.any(predictions == t[:, None], axis=1)

    return np.mean(accs)



