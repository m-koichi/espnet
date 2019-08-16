import scipy as sp
# from sklearn.datasets import fetch_mldata
import sklearn.base
import bhtsne
import matplotlib.pyplot as plot

import torch
import json
from dataset import SEDDataset
from transforms import Normalize
from torch.utils.data import DataLoader



import argparse
import logging
import os
import platform
import random
import subprocess
import sys
import json
# from sed_utils import make_batchset, CustomConverter

import numpy as np

from chainer.datasets import TransformDataset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator


# baseline modules
import sys
sys.path.append('./DCASE2019_task4/baseline')
from models.CNN import CNN
from models.RNN import BidirectionalGRU
import config as cfg
from models.CRNN import CRNN
from utils.Logger import LOG
from utils.utils import AverageMeterSet, weights_init, ManyHotEncoder, SaveBest
from evaluation_measures import compute_strong_metrics, segment_based_evaluation_df
from utils import ramps

import pdb


from dataset import SEDDataset
from transforms import Normalize, GaussianNoise, TimeWarp, FrequencyMask, TimeMask
from solver.mcd import MCDSolver
from solver.unet import UNet
from logger import Logger
from focal_loss import FocalLoss

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from scipy.signal import medfilt
import torch
import torch.nn as nn
import time
import pandas as pd
import re

from tqdm import tqdm
from datetime import datetime
from torchsummary import summary
# import ipdb

from dcase_util.data import DecisionEncoder
from dcase_util.data import ProbabilityEncoder

from distutils.util import strtobool

import matplotlib
matplotlib.use('Agg')


from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
from sklearn.metrics import explained_variance_score

#     return predictions
from torch.autograd import Function
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class Generator(nn.Module):
    def __init__(self, n_in_channel=1, activation="Relu", dropout=0, **kwargs):
        super(Generator, self).__init__()
        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        # if freq != 1:
        #     warnings.warn("Output shape is: {}".format((bs, frames, chan * freq)))
        #     x = x.permute(0, 2, 1, 3)
        #     x = x.contiguous().view(bs, frames, chan * freq)
        # else:
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        return x


class Classifier(nn.Module):
    def __init__(self, prob=0.5, lambd=1.0, n_class=10, attention=False, dropout=0,
                 n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(Classifier, self).__init__()

        self.prob = prob
        self.lambd = lambd

        self.rnn = BidirectionalGRU(64,
                                    n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)
        self.attention = attention
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            self.softmax = nn.Softmax(dim=-1)


    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)

        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong, weak

class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=100):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed
            # max_iter=self.max_iter
        )

if __name__ == "__main__":

    # read json data
    train_json = './data/train_aug/data_synthetic.json'
    train_weak_json = './data/train_aug/data_weak.json'
    valid_json = './data/validation/data_validation.json'
    #
    with open(train_json, 'rb') as train_json, \
            open(train_weak_json, 'rb') as train_weak_json, \
            open(valid_json, 'rb') as valid_json:
        train_json = json.load(train_json)['utts']
        train_weak_json = json.load(train_weak_json)['utts']
        # valid_json = json.load(valid_json)['utts']


    train_transforms = [Normalize()]
    test_transforms = [Normalize()]

    train_dataset = SEDDataset(train_json, transforms=train_transforms, pooling_time_ratio=8)
    train_weak_dataset = SEDDataset(train_weak_json, label_type='weak', transforms=train_transforms,
                                    pooling_time_ratio=8)
    # valid_dataset = SEDDataset(valid_json, transforms=test_transforms, pooling_time_ratio=8)

    train_synth_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    train_weak_loader = DataLoader(train_weak_dataset, batch_size=1, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    G_kwargs = {
        "n_in_channel": 1,
        "activation"  : "glu",
        "dropout"     : 0.5,
        "kernel_size" : 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 64, 64],
        "pooling"     : list(3 * ((2, 4),))
    }
    G_normal = Generator(**G_kwargs)
    G_adapt = Generator(**G_kwargs)

    # ipdb.set_trace()

    G_normal.cnn.load_state_dict(torch.load('./exp/2019_0527_model-mcd_rir-0_sa-0_pp-0_e-300/model/best.pth')['generator']['state_dict'])
    G_adapt.cnn.load_state_dict(torch.load('./exp/2019_0527_model-mcd_rir-0_sa-0_pp-0_e-300_ptr-8_l-BCE_no_adapt/model/best.pth')['generator']['state_dict'])
    # ipdb.set_trace()

    normal_feature =[]
    adapt_feature =[]

    G_normal = G_normal.cuda()
    G_adapt = G_adapt.cuda()

    G_normal.eval()
    G_adapt.eval()

    label = [0] * 200 + [1] * 200

    for i, (batch, _, _) in enumerate(train_synth_loader):
        with torch.no_grad():
            if i < 200:
                emb_n = G_normal(batch.cuda())
                emb_a = G_adapt(batch.cuda())
                normal_feature.append(emb_n[0].contiguous().view(-1).cpu().numpy())
                adapt_feature.append(emb_a[0].contiguous().view(-1).cpu().numpy())
    for i, (batch, _, _) in enumerate(train_weak_loader):
        with torch.no_grad():
            if i < 200:
                emb_n = G_normal(batch.cuda())
                emb_a = G_adapt(batch.cuda())
                normal_feature.append(emb_n[0].contiguous().view(-1).cpu().numpy())
                adapt_feature.append(emb_a[0].contiguous().view(-1).cpu().numpy())

    # import ipdb
    # ipdb.set_trace()
    decomp = TSNE(n_components=2)
    tsne = decomp.fit_transform(normal_feature)

    # bhtsne = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    # tsne = bhtsne.fit_transform(normal_feature)

    # xmin = tsne[:, 0].min()
    # xmax = tsne[:, 0].max()
    # ymin = tsne[:, 1].min()
    # ymax = tsne[:, 1].max()

    xmin = -20
    xmax = 20
    ymin = -20
    ymax = 20

    color = ['blue', 'red']

    plot.figure(figsize=(16, 12))
    for _y, _label in zip(tsne, label):
        plot.scatter(_y[0], _y[1], c=color[_label])
    plot.axis([xmin, xmax, ymin, ymax])
    plot.xlabel("component 0")
    plot.ylabel("component 1")
    plot.title("t-SNE visualization")
    plot.savefig("tsne_normal.png")

    # bhtsne = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    tsne = decomp.fit_transform(adapt_feature)

    xmin = tsne[:, 0].min()
    xmax = tsne[:, 0].max()
    ymin = tsne[:, 1].min()
    ymax = tsne[:, 1].max()

    plot.figure(figsize=(16, 12))
    for _y, _label in zip(tsne, label):
        plot.scatter(_y[0], _y[1], c=color[_label])
    plot.axis([xmin, xmax, ymin, ymax])
    plot.xlabel("component 0")
    plot.ylabel("component 1")
    plot.title("t-SNE visualization")
    plot.savefig("tsne_adapt.png")