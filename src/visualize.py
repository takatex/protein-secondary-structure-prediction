# -*- coding: utf-8 -*-

import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns

sns.set('talk', 'whitegrid', 'dark', font_scale=1.5,
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

def show_save(show, save, save_path):
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved in {save_path}')

    if show:
        plt.show()
    else:
        plt.close()

def seq_len_plot(seq_len, fontsize=20, labelsize=15,
                 show=True, save=False, save_path=''):
    seq_lens = np.zeros(700)
    c = collections.Counter(seq_len)
    keys, values = c.keys(), c.values()
    seq_lens[np.array(list(keys))-1] = list(values)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(np.arange(700), seq_lens)
    ax.vlines(seq_len.mean(), 0, max(list(values))+3, color='r', label='mean')
    ax.set_xlabel('sequence length', fontsize=fontsize)
    ax.set_ylabel('count', fontsize=fontsize)
    ax.set_ylim(0, max(list(values))+3)
    ax.legend()
    ax.tick_params(labelsize=labelsize)

    show_save(show, save, save_path)

def amino_rate_plot(amino_rate, fontsize=18, labelsize=15,
                    show=True, save=False, save_path=''):

    # sort amino rate
    values = sorted(amino_rate, key=lambda x: -x)
    idx = [np.where(amino_rate == v)[0][0] for v in values]
    keys = np.array(['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])[idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ['coral', 'orange', 'gold', 'yellowgreen',
              'lightgreen', 'mediumturquoise', 'c', 'royalblue']
    legend = [f'{k} : {v:.2f}' for k, v in zip(keys, values)]

    # pie chart
    patches, texts =  ax.pie(values, labels=keys, labeldistance=1.1,
                             counterclock=False, startangle=90,
                             colors=colors, wedgeprops={'linewidth': 2, 'edgecolor':"white"})

    # Create a circle for the center of the plot
    centre_circle = plt.Circle((0,0),0.65, color='white', fc='white',linewidth=1.25)
    plt.gcf().gca().add_artist(centre_circle)

    # option
    for t in texts:
      t.set_size(fontsize)
    ax.axis('equal')
    ax.legend(legend, bbox_to_anchor=(1.1, 0.5), loc="center right",
              borderaxespad=0., fontsize=fontsize)

    show_save(show, save, save_path)

def history_plot(history, title='', fontsize=20, labelsize=15,
                 show=True, save=False, save_path=''):
    train_loss, test_loss, acc = history

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 7))

    ax1.plot(train_loss)
    ax1.set_title(title, fontsize=fontsize)
    ax1.set_xlabel('epoch', fontsize=fontsize)
    ax1.set_ylabel('train loss', fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)

    ax2.plot(test_loss)
    ax2.set_title(title, fontsize=fontsize)
    ax2.set_xlabel('epoch', fontsize=fontsize)
    ax2.set_ylabel('test loss', fontsize=fontsize)
    ax2.tick_params(labelsize=labelsize)

    ax3.plot(acc)
    ax3.set_title(title, fontsize=fontsize)
    ax3.set_xlabel('epoch', fontsize=fontsize)
    ax3.set_ylabel('acc', fontsize=fontsize)
    ax3.tick_params(labelsize=labelsize)
    ax3.set_ylim(0, 1)

    show_save(show, save, save_path)

def acc_plot(df, fontsize=20, labelsize=15,
             show=True, save=False, save_path=''):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.boxplot(x='model', y='acc', data=df, ax=ax, palette="Set3")
    ax.set_xlabel('model')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1)

    show_save(show, save, save_path)

def acc_each_plot(df, fontsize=20, labelsize=15,
             show=True, save=False, save_path=''):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    keys = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    sns.boxplot(x='amino_acid', y='acc', data=df, order=keys, ax=ax, palette="Set3")
    ax.set_xlabel('amino acid')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1)

    show_save(show, save, save_path)

def loss_acc_plot(df_train_loss, df_test_loss, df_acc, title='',
                  show=True, save=False, save_path=''):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 7))
    sns.boxplot(x='model', y="train_loss",
                data=df_train_loss, palette="Set3", ax=ax1)
    ax1.set_title(title)
    sns.boxplot(x="model", y="test_loss",
                data=df_test_loss, palette="Set3", ax=ax2)
    ax2.set_title(title)

    sns.boxplot(x="model", y="acc",
                data=df_acc, palette="Set3", ax=ax3)
    ax3.set_title(title)

    show_save(show, save, save_path)
