#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


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
from logger import Logger
from focal_loss import FocalLoss

from torch.utils.data import DataLoader

from scipy.signal import medfilt
import torch
import torch.nn as nn
import time
import pandas as pd
import re

from tqdm import tqdm
from datetime import datetime
from torchsummary import summary

from dcase_util.data import DecisionEncoder
from dcase_util.data import ProbabilityEncoder

from distutils.util import strtobool

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def adjust_learning_rate(optimizer, rampup_value, rampdown_value):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def train(train_loader, model, optimizer, epoch, loss_function='BCE'):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    if loss_function == 'BCE':
        class_criterion = nn.BCELoss().to('cuda')
    elif loss_function == 'FocalLoss':
        class_criterion = FocalLoss(gamma=2).to('cuda')
    # consistency_criterion_strong = nn.MSELoss()
    # [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
    #     [class_criterion, consistency_criterion_strong])

    # meters = AverageMeterSet()

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    # rampup_length = len(train_loader) * cfg.n_epoch // 2
    for i, (batch_input, target, _) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        # if global_step < rampup_length:
        #     rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        # else:
        #     rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        # meters.update('lr', optimizer.param_groups[0]['lr'])

        batch_input = batch_input.to('cuda')
        target = target.to('cuda')

        # [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        # LOG.debug(batch_input.mean())
        # # Outputs
        # strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        # strong_pred_ema = strong_pred_ema.detach()
        # weak_pred_ema = weak_pred_ema.detach()

        # print(batch_input.shape)

        strong_pred, weak_pred = model(batch_input)

        # pdb.set_trace()
        loss = None
        # Weak BCE Loss
        # Take the max in the time axis
        # target_weak = target.max(-2)[0]

        # if weak_mask is not None:
        #     weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
        #     ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])
        #
        #     if i == 0:
        #         LOG.debug("target: {}".format(target.mean(-2)))
        #         LOG.debug("Target_weak: {}".format(target_weak))
        #         LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
        #         LOG.debug(weak_class_loss)
        #         LOG.debug("rampup_value: {}".format(rampup_value))
        #     meters.update('weak_class_loss', weak_class_loss.item())
        #
        #     meters.update('Weak EMA loss', ema_class_loss.item())
        #
        #     loss = weak_class_loss
        #
        # # Strong BCE loss
        # if strong_mask is not None:

        batch_size = strong_pred.size(0)
        # print(strong_pred.shape)
        strong_pred = strong_pred.permute(0, 2, 1).contiguous().view(batch_size, -1, 1).repeat(1, 1, 8).view(batch_size, -1, 10)
        # print(strong_pred.shape)
        # print(target.shape)
        strong_class_loss = class_criterion(strong_pred, target)
        # meters.update('Strong loss', strong_class_loss.item())

        # strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
        # meters.update('Strong EMA loss', strong_ema_class_loss.item())
        if loss is not None:
            loss += strong_class_loss
        else:
            loss = strong_class_loss

        # # Teacher-student consistency cost
        # if ema_model is not None:
        #
        #     consistency_cost = cfg.max_consistency_cost * rampup_value
        #     meters.update('Consistency weight', consistency_cost)
        #     # Take only the consistence with weak and unlabel
        #     consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred,
        #                                                                               strong_pred_ema)
        #     meters.update('Consistency strong', consistency_loss_strong.item())
        #     if loss is not None:
        #         loss += consistency_loss_strong
        #     else:
        #         loss = consistency_loss_strong
        #
        #     meters.update('Consistency weight', consistency_cost)
        #     # Take only the consistence with weak and unlabel
        #     consistency_loss_weak = consistency_cost * consistency_criterion_strong(weak_pred, weak_pred_ema)
        #     meters.update('Consistency weak', consistency_loss_weak.item())
        #     if loss is not None:
        #         loss += consistency_loss_weak
        #     else:
        #         loss = consistency_loss_weak
        #
        # assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        # assert not loss.item() < 0, 'Loss problem, cannot be negative'
        # meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        # if ema_model is not None:
        #     update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start


def train_strong_only(strong_loader, model, optimizer, epoch):
    class_criterion = nn.BCELoss().to('cuda')

    # meters = AverageMeterSet()
    # meters.update('lr', optimizer.param_groups[0]['lr'])

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, (batch_input, target, _) in enumerate(strong_loader):
        # [batch_input, target] = to_cuda_if_available([batch_input, target])
        # LOG.debug(batch_input.mean())
        batch_input = batch_input.to('cuda')
        target = target.to('cuda')

        strong_pred, weak_pred = model(batch_input)
        loss = 0

        # if strong_mask is not None:
        # Strong BCE loss
        strong_class_loss = class_criterion(strong_pred, target)
        # meters.update('Strong loss', strong_class_loss.item())

        loss += strong_class_loss

        # assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        # assert not loss.item() < 0, 'Loss problem, cannot be negative'
        # meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - start

    # LOG.info(
    #         'Epoch: {}\t'
    #         'Time {:.2f}\t'
    #         '{meters}'.format(
    #                 epoch, epoch_time, meters=meters))
    #

def train_strong_weak(strong_loader, weak_loader, model, optimizer, epoch, logger, loss_function='BCE'):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    if loss_function == 'BCE':
        class_criterion = nn.BCELoss().to('cuda')
    elif loss_function == 'FocalLoss':
        class_criterion = FocalLoss(gamma=2).to('cuda')
    # consistency_criterion_strong = nn.MSELoss()
    # [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
    #     [class_criterion, consistency_criterion_strong])

    # meters = AverageMeterSet()

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(strong_loader) * cfg.n_epoch // 2
    for i, ((s_batch_input, s_target, _), (w_batch_input, w_target, _)) in \
            enumerate(zip(strong_loader, weak_loader)):
        # global_step = epoch * len(strong_loader) + i
        # if global_step < rampup_length:
        #     rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        # else:
        #     rampup_value = 1.0


        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        # meters.update('lr', optimizer.param_groups[0]['lr'])

        s_batch_input = s_batch_input.to('cuda')
        s_target = s_target.to('cuda')
        w_batch_input = w_batch_input.to('cuda')
        w_target = w_target.to('cuda')

        # [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        # LOG.debug(batch_input.mean())
        # # Outputs
        # strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        # strong_pred_ema = strong_pred_ema.detach()
        # weak_pred_ema = weak_pred_ema.detach()

        # print(batch_input.shape)

        s_strong_pred, s_weak_pred = model(s_batch_input)
        w_strong_pred, w_weak_pred = model(w_batch_input)

        # pdb.set_trace()
        loss = 0
        # Weak BCE Loss
        # Take the max in the time axis
        # target_weak = target.max(-2)[0]

        # if weak_mask is not None:
        #     weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
        #     ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])
        #
        #     if i == 0:
        #         LOG.debug("target: {}".format(target.mean(-2)))
        #         LOG.debug("Target_weak: {}".format(target_weak))
        #         LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
        #         LOG.debug(weak_class_loss)
        #         LOG.debug("rampup_value: {}".format(rampup_value))
        #     meters.update('weak_class_loss', weak_class_loss.item())
        #
        #     meters.update('Weak EMA loss', ema_class_loss.item())
        #
        #     loss = weak_class_loss
        #
        # # Strong BCE loss
        # if strong_mask is not None:

        # batch_size = s_strong_pred.size(0)
        # # print(strong_pred.shape)
        # s_strong_pred = s_strong_pred.permute(0, 2, 1).contiguous().view(batch_size, -1, 1).repeat(1, 1, 8).view(
        #     batch_size, -1, 10)
        # print(strong_pred.shape)
        # print(target.shape)
        # pdb.set_trace()
        strong_class_loss = class_criterion(s_strong_pred, s_target)
        # pdb.set_trace()
        # w_target = w_target.max(-2)[0]
        weak_class_loss = class_criterion(w_weak_pred, w_target)
        # meters.update('Strong loss', strong_class_loss.item())

        # strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
        # meters.update('Strong EMA loss', strong_ema_class_loss.item())
        # if loss is not None:
        #     loss += strong_class_loss + weak_class_loss
        # else:
        loss = strong_class_loss + weak_class_loss

        # # Teacher-student consistency cost
        # if ema_model is not None:
        #
        #     consistency_cost = cfg.max_consistency_cost * rampup_value
        #     meters.update('Consistency weight', consistency_cost)
        #     # Take only the consistence with weak and unlabel
        #     consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred,
        #                                                                               strong_pred_ema)
        #     meters.update('Consistency strong', consistency_loss_strong.item())
        #     if loss is not None:
        #         loss += consistency_loss_strong
        #     else:
        #         loss = consistency_loss_strong
        #
        #     meters.update('Consistency weight', consistency_cost)
        #     # Take only the consistence with weak and unlabel
        #     consistency_loss_weak = consistency_cost * consistency_criterion_strong(weak_pred, weak_pred_ema)
        #     meters.update('Consistency weak', consistency_loss_weak.item())
        #     if loss is not None:
        #         loss += consistency_loss_weak
        #     else:
        #         loss = consistency_loss_weak
        #
        # assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        # assert not loss.item() < 0, 'Loss problem, cannot be negative'
        # meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # global_step += 1
        # if ema_model is not None:
        #     update_ema_variables(model, ema_model, 0.999, global_step)

    # epoch_time = time.time() - start
    logger.scalar_summary('strong class loss', strong_class_loss.item(), epoch)
    logger.scalar_summary('weak class loss', weak_class_loss.item(), epoch)
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tstrong class loss: {strong_class_loss.item()}')
    LOG.info(f'\tweak class loss: {weak_class_loss.item()}')


# LOG.info(
#     'Epoch: {}\t'
#     'Time {:.2f}\t'
#     '{meters}'.format(
#         epoch, epoch_time, meters=meters))


def get_batch_predictions(model, data_loader, decoder, post_processing=False, save_predictions=None):
    prediction_df = pd.DataFrame()
    for batch_idx, (batch_input, _, data_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()

        pred_strong, _ = model(batch_input)
        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)
        if post_processing:
            for i in range(pred_strong.shape[0]):
                pred_strong[i] = median_filt_1d(pred_strong[i])
                pred_strong[i] = fill_up_gap(pred_strong[i])
                pred_strong[i] = remove_short_duration(pred_strong[i])

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)

            # if batch_idx == 0:
            #     LOG.debug("predictions: \n{}".format(pred))
            #     LOG.debug("predictions strong: \n{}".format(pred_strong))
            #     prediction_df = pred.copy()
            # else:


    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")
    return prediction_df


# def compute_strong_metrics(predictions, data_loader):
def median_filt_1d(event_roll, filt_span=3):
    """FUNCTION TO APPLY MEDIAN FILTER
    ARGS:
    --
    event_roll: event roll [T,C]
    filt_span: median filter span(integer odd scalar)
    RETURN:
    --
    event_roll : median filter applied event roll [T,C]
    """
    if len(event_roll.shape) == 1:
        event_roll = medfilt(event_roll, filt_span)
    else:
        for i in range(event_roll.shape[1]):
            event_roll[:, i] = medfilt(event_roll[:, i], filt_span)

    return event_roll


def fill_up_gap(event_roll, accept_gap=1):
    """FUNCTION TO PERFORM FILL UP GAPS
    ARGS:
    --
    event_roll: event roll [T,C]
    accept_gap: number of accept gap to fill up (integer scalar)
    RETURN:
    --
    event_roll: processed event roll [T,C]
    """
    num_classes = event_roll.shape[1]
    event_roll_ = np.append(
            np.append(
                    np.zeros((1, num_classes)),
                    event_roll, axis=0),
            np.zeros((1, num_classes)),
            axis=0)
    aux_event_roll = np.diff(event_roll_, axis=0)

    for i in range(event_roll.shape[1]):
        onsets = np.where(aux_event_roll[:, i] == 1)[0]
        offsets = np.where(aux_event_roll[:, i] == -1)[0]
        for j in range(1, onsets.shape[0]):
            if onsets[j] - offsets[j - 1] <= accept_gap:
                event_roll[offsets[j - 1]:onsets[j], i] = 1

    return event_roll


def remove_short_duration(event_roll, reject_duration=3):
    """Remove short duration
    ARGS:
    --
    event_roll: event roll [T,C]
    reject_duration: number of duration to reject as short section (int or list)
    RETURN:
    --
    event_roll: processed event roll [T,C]
    """
    num_classes = event_roll.shape[1]
    event_roll_ = np.append(
            np.append(
                    np.zeros((1, num_classes)),
                    event_roll, axis=0),
            np.zeros((1, num_classes)),
            axis=0)
    aux_event_roll = np.diff(event_roll_, axis=0)

    for i in range(event_roll.shape[1]):
        onsets = np.where(aux_event_roll[:, i] == 1)[0]
        offsets = np.where(aux_event_roll[:, i] == -1)[0]
        for j in range(onsets.shape[0]):
            if isinstance(reject_duration, int):
                if onsets[j] - offsets[j] <= reject_duration:
                    event_roll[offsets[j]:onsets[j], i] = 0
            elif isinstance(reject_duration, list):
                if onsets[j] - offsets[j] <= reject_duration[i]:
                    event_roll[offsets[j]:onsets[j], i] = 0

    return event_roll


def save_args(args, dest_dir, name='config.yml'):
    import yaml
    print(yaml.dump(vars(args)))
    with open(os.path.join(dest_dir, name), 'w') as f:
        f.write(yaml.dump(vars(args)))

# def post_processing(predictions, window_length=5):
#
#     post_processed = 0
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

def main(args):
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--outdir', type=str,  default='../exp/results',
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--debugdir', type=str,
                        help='Output directory for debugging')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    # task related
    parser.add_argument('--train-json', type=str, default='./data/train_aug/data_synthetic.json',
                        help='Filename of train label data (json)')
    parser.add_argument('--train-weak-json', type=str, default='./data/train_aug/data_weak.json',
                        help='Filename of train weak label data (json)')
    parser.add_argument('--valid-json', type=str, default='./data/validation/data_validation.json',
                        help='Filename of validation label data (json)')
    parser.add_argument('--valid-meta', type=str, default='./DCASE2019_task4/dataset/metadata/validation/validation.csv',
                        help='Metadata of validation data (csv)')
    # network architecture
    # encoder
    # parser.add_argument('--num-spkrs', default=1, type=int,
    #                     choices=[1, 2],
    #                     help='Number of speakers in the speech.')
    # parser.add_argument('--etype', default='blstmp', type=str,
    #                     choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
    #                              'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
    #                     help='Type of encoder network architecture')
    # parser.add_argument('--elayers-sd', default=4, type=int,
    #                     help='Number of encoder layers for speaker differentiate part. (multi-speaker asr mode only)')
    # parser.add_argument('--elayers', default=4, type=int,
    #                     help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
    # parser.add_argument('--eunits', '-u', default=300, type=int,
    #                     help='Number of encoder hidden units')
    # parser.add_argument('--eprojs', default=320, type=int,
    #                     help='Number of encoder projection units')
    # parser.add_argument('--subsample', default=1, type=str,
    #                     help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
    #                          'every y frame at 2nd layer etc.')
    # loss
    # parser.add_argument('--ctc_type', default='warpctc', type=str,
    #                     choices=['builtin', 'warpctc'],
    #                     help='Type of CTC implementation to calculate loss.')
    # # attention
    # parser.add_argument('--atype', default='dot', type=str,
    #                     choices=['noatt', 'dot', 'add', 'location', 'coverage',
    #                              'coverage_location', 'location2d', 'location_recurrent',
    #                              'multi_head_dot', 'multi_head_add', 'multi_head_loc',
    #                              'multi_head_multi_res_loc'],
    #                     help='Type of attention architecture')
    # parser.add_argument('--adim', default=320, type=int,
    #                     help='Number of attention transformation dimensions')
    # parser.add_argument('--awin', default=5, type=int,
    #                     help='Window size for location2d attention')
    # parser.add_argument('--aheads', default=4, type=int,
    #                     help='Number of heads for multi head attention')
    # parser.add_argument('--aconv-chans', default=-1, type=int,
    #                     help='Number of attention convolution channels \
    #                     (negative value indicates no location-aware attention)')
    # parser.add_argument('--aconv-filts', default=100, type=int,
    #                     help='Number of attention convolution filters \
    #                     (negative value indicates no location-aware attention)')
    # parser.add_argument('--spa', action='store_true',
    #                     help='Enable speaker parallel attention.')
    # decoder
    # parser.add_argument('--dtype', default='lstm', type=str,
    #                     choices=['lstm', 'gru'],
    #                     help='Type of decoder network architecture')
    # parser.add_argument('--dlayers', default=1, type=int,
    #                     help='Number of decoder layers')
    # parser.add_argument('--dunits', default=320, type=int,
    #                     help='Number of decoder hidden units')
    # parser.add_argument('--mtlalpha', default=0.5, type=float,
    #                     help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
    # parser.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
    #                     help='Apply label smoothing with a specified distribution type')
    # parser.add_argument('--lsm-weight', default=0.0, type=float,
    #                     help='Label smoothing weight')
    # parser.add_argument('--sampling-probability', default=0.0, type=float,
    #                     help='Ratio of predicted labels fed back to decoder')
    # # recognition options to compute CER/WER
    # parser.add_argument('--report-cer', default=False, action='store_true',
    #                     help='Compute CER on development set')
    # parser.add_argument('--report-wer', default=False, action='store_true',
    #                     help='Compute WER on development set')
    # parser.add_argument('--nbest', type=int, default=1,
    #                     help='Output N-best hypotheses')
    # parser.add_argument('--beam-size', type=int, default=4,
    #                     help='Beam size')
    # parser.add_argument('--penalty', default=0.0, type=float,
    #                     help='Incertion penalty')
    # parser.add_argument('--maxlenratio', default=0.0, type=float,
    #                     help="""Input length ratio to obtain max output length.
    #                     If maxlenratio=0.0 (default), it uses a end-detect function
    #                     to automatically find maximum hypothesis lengths""")
    # parser.add_argument('--minlenratio', default=0.0, type=float,
    #                     help='Input length ratio to obtain min output length')
    # parser.add_argument('--ctc-weight', default=0.3, type=float,
    #                     help='CTC weight in joint decoding')
    # parser.add_argument('--rnnlm', type=str, default=None,
    #                     help='RNNLM model file to read')
    # parser.add_argument('--rnnlm-conf', type=str, default=None,
    #                     help='RNNLM model config file to read')
    # parser.add_argument('--lm-weight', default=0.1, type=float,
    #                     help='RNNLM weight.')
    # parser.add_argument('--sym-space', default='<space>', type=str,
    #                     help='Space symbol')
    # parser.add_argument('--sym-blank', default='<blank>', type=str,
    #                     help='Blank symbol')
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                        help='Dropout rate for the decoder')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch-size', '-b', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # optimization related
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam'],
                        help='Optimizer')
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help='Weight decay ratio')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/acc', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=3, type=int,
                        help='Number of samples of attention to be saved')
    parser.add_argument('--use-rir-augmentation', default=False, type=strtobool)
    parser.add_argument('--use-specaugment', default=False, type=strtobool)
    parser.add_argument('--use-post-processing', default=False, type=strtobool)
    parser.add_argument('--model', default='crnn_baseline_feature', type=str)
    parser.add_argument('--pooling-time-ratio', default=8, type=int)
    parser.add_argument('--loss-function', default='BCE', type=str,
                        choices=['BCE', 'FocalLoss'],
                        help='Type of loss function')
    # transfer learning related
    # parser.add_argument('--sed-model', default=False, nargs='?',
    #                     help='Pre-trained SED model')
    # parser.add_argument('--mt-model', default=False, nargs='?',
    #                     help='Pre-trained MT model')
    args = parser.parse_args(args)

    exp_name = os.path.join('exp', datetime.now().strftime("%Y_%m%d_%H%M%S"))
    exp_name = f'exp/{datetime.now().strftime("%Y_%m%d")}_model-{args.model}_rir-{args.use_specaugment}' \
               f'_sa-{args.use_specaugment}_pp-{args.use_post_processing}_e-{args.epochs}' \
               f'_ptr-{args.pooling_time_ratio}_l-{args.loss_function}'
    os.makedirs(os.path.join(exp_name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'log'), exist_ok=True)
    save_args(args, exp_name)
    logger = Logger(os.path.join(exp_name, 'log'))


    # read json data
    if args.use_rir_augmentation:
        train_json = './data/train_aug/data_synthetic.json'
        train_weak_json = './data/train_aug/data_weak.json'
        valid_json = './data/validation/data_validation.json'
    else:
        train_json = './data/train/data_synthetic.json'
        train_weak_json = './data/train/data_weak.json'
        valid_json = './data/validation/data_validation.json'

    with open(train_json, 'rb') as train_json, \
         open(train_weak_json, 'rb') as train_weak_json, \
         open(valid_json, 'rb') as valid_json:
        train_json = json.load(train_json)['utts']
        train_weak_json = json.load(train_weak_json)['utts']
        valid_json = json.load(valid_json)['utts']

    if args.use_specaugment:
        train_transforms = [Normalize(), TimeWarp(), FrequencyMask(), TimeMask()]
        test_transforms = [Normalize()]
    else:
        train_transforms = [Normalize()]
        test_transforms = [Normalize()]
    train_dataset = SEDDataset(train_json, transforms=train_transforms, pooling_time_ratio=args.pooling_time_ratio)
    train_weak_dataset = SEDDataset(train_weak_json, label_type='weak', transforms=train_transforms, pooling_time_ratio=args.pooling_time_ratio)
    valid_dataset = SEDDataset(valid_json, transforms=test_transforms, pooling_time_ratio=args.pooling_time_ratio)

    train_synth_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    validation_df = pd.read_csv(args.valid_meta, header=0, sep="\t")

    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=864 / args.pooling_time_ratio)

    # build model
    crnn_kwargs = cfg.crnn_kwargs
    if args.pooling_time_ratio == 1:
        crnn_kwargs['pooling'] = list(3 * ((1, 4),))
    elif args.pooling_time_ratio == 8:
        pass
    else:
        raise ValueError
    crnn = CRNN(**crnn_kwargs)
    crnn.apply(weights_init)
    crnn = crnn.to('cuda')
    # summary(crnn, (1, 864, 64))
    # pdb.set_trace()
    # crnn_ema = CRNN(**crnn_kwargs)

    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)

    if args.model == 'mcd':
        # For MCD
        G_kwargs = {
             "n_in_channel": 1,
             "activation"  : "glu",
             "dropout"     : 0.5,
             "kernel_size" : 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 64, 64],
             "pooling"     : list(3 * ((2, 4),))
        }
        if args.pooling_time_ratio == 1:
            G_kwargs['pooling'] = list(3 * ((1, 4),))
        F_kwargs = {
            "n_class"     : 10, "attention": True, "n_RNN_cell": 64,
                        "n_layers_RNN": 2,
                        "dropout"     : 0.5
        }
        G = Generator(**G_kwargs)
        F1 = Classifier(**F_kwargs)
        F2 = Classifier(**F_kwargs)

        G.apply(weights_init)
        F1.apply(weights_init)
        F2.apply(weights_init)

        optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), **optim_kwargs)
        optimizer_f = torch.optim.Adam(filter(lambda p: p.requires_grad, list(F1.parameters()) + list(F2.parameters())), **optim_kwargs)

        MCD = MCDSolver(exp_name=exp_name,
                        source_loader=train_synth_loader,
                        target_loader=train_weak_loader,
                        generator=G,
                        classifier1=F1,
                        classifier2=F2,
                        optimizer_g=optimizer_g,
                        optimizer_f=optimizer_f,
                        num_k=4,
                        num_multiply_d_loss=1)

        state = {
            'generator'             : {"name"      : G.__class__.__name__,
                                'args'      : '',
                                "kwargs"    : G_kwargs,
                                'state_dict': G.state_dict()},
            'classifier1'       :{"name"      : F1.__class__.__name__,
                                'args'      : '',
                                "kwargs"    : F_kwargs,
                                'state_dict': F1.state_dict()},
            'classifier2'       :{"name"      : F2.__class__.__name__,
                                'args'      : '',
                                "kwargs"    : F_kwargs,
                                'state_dict': F2.state_dict()},
            'optimizer_g'       : {"name"      : optimizer_g.__class__.__name__,
                                'args'      : '',
                                "kwargs"    : optim_kwargs,
                                'state_dict': optimizer_g.state_dict()},
            'optimizer_f'       : {"name"      : optimizer_f.__class__.__name__,
                                'args'      : '',
                                "kwargs"    : optim_kwargs,
                                'state_dict': optimizer_f.state_dict()},
            "pooling_time_ratio": args.pooling_time_ratio,
            'scaler'            : None,
            "many_hot_encoder"  : many_hot_encoder.state_dict()
        }
        save_best_eb = SaveBest("sup")
        save_best_sb = SaveBest("sup")
        best_event_epoch = 0
        best_event_f1 = 0
        best_segment_epoch = 0
        best_segment_f1 = 0

        for epoch in range(args.epochs):
            MCD.train()
            # TODO: post-process, save best
            # MCD.test(validation_df, valid_loader, many_hot_encoder, i)

            MCD.set_eval()
        # if epoch > 50:

            with torch.no_grad():
                predictions = MCD.get_batch_predictions(valid_loader, many_hot_encoder, epoch)
                valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio)
                valid_segments_metric = segment_based_evaluation_df(validation_df, predictions, time_resolution=float(args.pooling_time_ratio))
                # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
            state['generator']['state_dict'] = MCD.G.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer_g']['state_dict'] = MCD.optimizer_g.state_dict()
            state['classifier1']['state_dict'] = MCD.F1.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer_f']['state_dict'] = MCD.optimizer_f.state_dict()
            state['classifier2']['state_dict'] = MCD.F2.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer_f']['state_dict'] = MCD.optimizer_f.state_dict()
            state['epoch'] = epoch
            state['valid_metric'] = valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'epoch_{epoch + 1}.pth'))

            # pdb.set_trace()
            global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            # global_valid = global_valid + np.mean(weak_metric)
            if save_best_eb.apply(global_valid):
                best_epoch = epoch + 1
                best_score = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                torch.save(state, model_fname)

            # For debug
            segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            if save_best_sb.apply(segment_valid):
                best_epoch = epoch + 1
                best_score = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                torch.save(state, model_fname)

        if cfg.save_best:
            model_fname = os.path.join(exp_name, 'model', "best.pth")
            state = torch.load(model_fname)
            LOG.info("testing model: {}".format(model_fname))

        LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
        LOG.info("Event-based: best macro-f1 score: {}".format(best_event_score))
        LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
        LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_score))

        pdb.set_trace()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == '2':
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).decode().strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    state = {
        'model'             : {"name"      : crnn.__class__.__name__,
                               'args'      : '',
                               "kwargs"    : crnn_kwargs,
                               'state_dict': crnn.state_dict()},
        'optimizer'         : {"name"      : optimizer.__class__.__name__,
                               'args'      : '',
                               "kwargs"    : optim_kwargs,
                               'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": args.pooling_time_ratio,
        'scaler'            : None,
        "many_hot_encoder"  : many_hot_encoder.state_dict()
    }
    save_best_eb = SaveBest("sup")
    save_best_sb = SaveBest("sup")
    best_event_epoch = 0
    best_event_f1 = 0
    best_segment_epoch = 0
    best_segment_f1 = 0

    # model training
    for epoch in tqdm(range(args.epochs)):
        crnn = crnn.train()
        # train(train_loader, crnn, optimizer, epoch)
        # train_strong_only(train_synth_loader, crnn, optimizer, epoch)
        train_strong_weak(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger, args.loss_function)

        crnn = crnn.eval()
        # if epoch > 50:

        with torch.no_grad():
            predictions = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
                                                post_processing=args.use_post_processing,
                                            save_predictions=os.path.join(exp_name, 'predictions', f'result_{epoch}.csv'))
            valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio)
            valid_segments_metric = segment_based_evaluation_df(validation_df, predictions, time_resolution=float(args.pooling_time_ratio))
            # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
        state['model']['state_dict'] = crnn.state_dict()
        # state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        torch.save(state, os.path.join(exp_name, 'model', f'epoch_{epoch + 1}.pth'))

        # pdb.set_trace()
        global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        # global_valid = global_valid + np.mean(weak_metric)
        if save_best_eb.apply(global_valid):
            best_epoch = epoch + 1
            best_score = global_valid
            model_fname = os.path.join(exp_name, 'model', "best.pth")
            torch.save(state, model_fname)

        # For debug
        segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        if save_best_sb.apply(segment_valid):
            best_epoch = epoch + 1
            best_score = global_valid
            model_fname = os.path.join(exp_name, 'model', "best.pth")
            torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(exp_name, 'model', "best.pth")
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))

    LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
    LOG.info("Event-based: best macro-f1 score: {}".format(best_event_score))
    LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
    LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_score))

    # load dictionary for debug log
    # if args.dict is not None:
    #     with open(args.dict, 'rb') as f:
    #         dictionary = f.readlines()
    #     char_list = [entry.decode('utf-8').split(' ')[0]
    #                  for entry in dictionary]
    #     char_list.insert(0, '<blank>')
    #     char_list.append('<eos>')
    #     args.char_list = char_list
    # else:
    #     args.char_list = None



if __name__ == '__main__':
    main(sys.argv[1:])
