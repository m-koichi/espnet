import os
import shutil
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from transforms import GaussianNoise, FrequencyMask


def data_augmentation(X):
    pass


def unlabel_augment(ub, K, augment):
    ubk = []
    for k in K:
        ubk.append(augment(ub))
    return ubk


def label_guessing(model, ubk):
    model.eval()
    pred = model(ubk)
    qb =  torch.mean(pred)
    return qb


def sharpening(p, T):
    pt = p ** (1 / T)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    return targets_u


def shuffle(X, U):
    indices = np.arange(len(X) + len(U))
    np.random.shuffle(indices)


#
# def mixup(x, u, p, q, alpha=1.0):
#     if 1.0 > alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1.0
#     lam = max(lam, 1 - lam)
#
#     mixed_data = lam * x + (1 - lam) * u
#     mixed_label = lam * p + (1 - lam) * q
#     return mixed_data, mixed_label


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def loss():
    pass


def augment_fn(data, transforms=[GaussianNoise(), FrequencyMask()]):
    for i in range(data.shape[0]):
        for transform in transforms:
            data[i] = transform(data[i])
    return data

#
# def mixmatch(model, X, U, T=0.5, K=2, alpha=0.75):
#     for b in X.size()[0]:
#         X[b] = augment(X[b])
#         ubk = unlabel_augment(U[b], K=K)
#         qb = label_guessing(model, ubk)
#         qb = sharpening(qb, T=T)
#     W = shuffle(torch.concat(X, U), torch.concat(p, q))
#     X = mixup(X, W[::2], alpha=alpha)
#     U = mixup(W, W[1::2], alpha=alpha)
#     return X, U, p, q


def sharpen(x, T):
    # import ipdb
    # ipdb.set_trace()
    temp = x**(1/T)
    return temp / temp.sum(dim=1, keepdim=True)

def mixup(x, u, p, q, alpha=1.0):
    if 1.0 > alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    lam = max(lam, 1 - lam)

    mixed_data = lam * x + (1 - lam) * u
    mixed_label = np.logical_or(p, q) * 1
    return mixed_data, mixed_label


def mixmatch(x, y, u, model, T=0.5, K=2, alpha=0.75, strong=True):
    """

    :param x:  numpy batch labeled data
    :param y:  numpy batch target
    :param u:  numpy batch unlabeled data
    :param model:  model
    :param augment_fn: data augmentation
    :param T:
    :param K:
    :param alpha:
    :return:
    """
    xb = augment_fn(x)
    ub = [augment_fn(u) for _ in range(K)]
    if strong:
        qb = sharpen(sum(map(lambda i: model(torch.from_numpy(i).cuda())[0], ub)) / K, T)
    else:
        qb = sharpen(sum(map(lambda i: model(torch.from_numpy(i).cuda())[1], ub)) / K, T)
    qb = qb.cpu().detach().numpy()
    Ux = np.concatenate(ub, axis=0)
    Uy = np.concatenate([qb for _ in range(K)], axis=0)
    indices = np.random.permutation(len(xb) + len(Ux))
    Wx = np.concatenate([xb, Ux], axis=0)[indices]
    Wy = np.concatenate([y, Uy], axis=0)[indices]
    X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    return X, U, p, q


class MixMatchLoss(nn.Module):
    def __init__(self, num_classes=10, lambda_u=100):
        super(MixMatchLoss, self).__init__()
        self.lambda_u = lambda_u
        self.ent_loss = nn.BCELoss().cuda()
        self.l2loss = nn.MSELoss().cuda()
        self.L = num_classes

    def forward(self, pred_x, pred_u, p, q):
        # import ipdb
        # ipdb.set_trace()
        Lx = self.ent_loss(pred_x, p)
        Lu = (1 / self.L) * self.l2loss(pred_u, q)
        return Lx + self.lambda_u * Lu
