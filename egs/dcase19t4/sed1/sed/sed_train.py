#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import json
import logging
import os
import platform
import random
import subprocess
# baseline modules
import sys

import numpy as np

# from sed_utils import make_batchset, CustomConverter

sys.path.append('./DCASE2019_task4/baseline')
from models.CNN import CNN
from models.RNN import BidirectionalGRU
import config as cfg
from utils.Logger import LOG
from utils.utils import weights_init, ManyHotEncoder, SaveBest
from evaluation_measures import compute_strong_metrics, segment_based_evaluation_df
from utils import ramps

import pdb

from dataset import SEDDataset
from transforms import Normalize, ApplyLog, GaussianNoise, FrequencyMask
from solver.mcd import MCDSolver
from solver.unet import UNet1D, BCEDiceLoss
# from solver.CNN import CNN
# from solver.RNN import RNN
from solver.CRNN import CRNN

from logger import Logger
from focal_loss import FocalLoss

from torch.utils.data import DataLoader, ConcatDataset
import torch.backends.cudnn as cudnn

from scipy.signal import medfilt
import torch
import torch.nn as nn
import time
import pandas as pd
import re

from tqdm import tqdm
from datetime import datetime
import pickle
import ipdb

from dcase_util.data import ProbabilityEncoder

from distutils.util import strtobool
from solver.mix_match import mixmatch, MixMatchLoss
import adabound
from torch.optim.lr_scheduler import StepLR

from model_tuning import search_best_threshold, search_best_median, search_best_accept_gap, \
    search_best_remove_short_duration, show_best

global_step = 0


def lr_test(model, train_loader, criterion, optimizer, step=0.1):
    model = model.cuda()
    criterion = criterion.cuda()
    for data, target, _ in train_loader:
        pred_strong, pred_weak = model(data.cuda())
        loss = criterion.cuda()
        scheduler.step()
        print(loss.item())



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
        strong_pred = strong_pred.permute(0, 2, 1).contiguous().view(batch_size, -1, 1).repeat(1, 1, 8).view(batch_size,
                                                                                                             -1, 10)
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

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    avg_strong_loss = 0
    avg_weak_loss = 0

    for i, ((s_batch_input, s_target, s_data), (w_batch_input, w_target, w_data)) in \
            enumerate(zip(strong_loader, weak_loader)):
        s_batch_input, s_target = s_batch_input.to('cuda'), s_target.cuda()
        w_batch_input, w_target = w_batch_input.to('cuda'), w_target.cuda()

        s_strong_pred, s_weak_pred = model(s_batch_input)
        w_strong_pred, w_weak_pred = model(w_batch_input)

        strong_class_loss = class_criterion(s_strong_pred, s_target)
        weak_class_loss = class_criterion(w_weak_pred, w_target)
        # meters.update('Strong loss', strong_class_loss.item())
        logger.scalar_summary('train_strong_loss', strong_class_loss.item(), epoch)
        logger.scalar_summary('train_weak_loss', weak_class_loss.item(), epoch)

        loss = strong_class_loss + weak_class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        # meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))

    # epoch_time = time.time() - start
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train_one_step(strong_loader, weak_loader, model, optimizer, logger, loss_function='BCE', iterations=10000,
                   log_interval=100,
                   valid_loader=None,
                   validation_df=None,
                   many_hot_encoder=None,
                   args=None,
                   exp_name=None,
                   state=None,
                   save_best_eb=None,
                   lr_scheduler=None):
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

    best_iterations = 0
    best_f1 = 0

    global global_step
    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(strong_loader) * cfg.n_epoch // 2
    avg_strong_loss = 0
    avg_weak_loss = 0

    sample_rate = 44100 if args.n_frames == 864 else 16000
    hop_length = 511 if args.n_frames == 864 else 320

    strong_iter = iter(strong_loader)
    weak_iter = iter(weak_loader)

    for i in range(1, iterations + 1):
        try:
            strong_sample, strong_target, _ = next(strong_iter)
        except:
            strong_iter = iter(strong_loader)
            strong_sample, strong_target, _ = next(strong_iter)
        try:
            weak_sample, weak_target, _ = next(weak_iter)
        except:
            weak_iter = iter(weak_loader)
            weak_sample, weak_target, _ = next(weak_iter)

        strong_sample = strong_sample.to('cuda')
        strong_target = strong_target.to('cuda')
        weak_sample = weak_sample.to('cuda')
        weak_target = weak_target.to('cuda')

        pred_strong, pred_weak = model(strong_sample)
        strong_class_loss = class_criterion(pred_strong, strong_target)

        pred_strong, pred_weak = model(weak_sample)
        weak_class_loss = class_criterion(pred_weak, weak_target)

        global_step += 1
        logger.scalar_summary('train_strong_loss', strong_class_loss.item(), global_step)
        logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)

        loss = strong_class_loss + weak_class_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # predictions = get_batch_predictions(crnn, sad_valid_loader, many_hot_encoder.decode_strong,
                #                                     post_processing=args.use_post_processing,
                #                                     save_predictions=os.path.join(exp_name, 'predictions',
                #                                                                   f'result_{epoch}.csv'))
                # valid_events_metric = compute_strong_metrics(predictions, synth_df, args.pooling_time_ratio)

                predictions = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'result_{i}.csv'))
                valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                             sample_rate=sample_rate, hop_length=hop_length)
                valid_segments_metric = segment_based_evaluation_df(validation_df, predictions,
                                                                    time_resolution=float(args.pooling_time_ratio))
                # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
            state['model']['state_dict'] = model.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer']['state_dict'] = optimizer.state_dict()
            state['iterations'] = i
            state['valid_metric'] = valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'iteration_{i}.pth'))

            global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

            if save_best_eb.apply(global_valid):
                best_iterations = i
                best_f1 = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                torch.save(state, model_fname)
            model.train()

    return best_iterations, best_f1


def train_one_step_ema(strong_loader, weak_loader, unlabel_loader, model, ema_model, optimizer, logger,
                       loss_function='BCE', iterations=10000,
                       log_interval=100,
                       valid_loader=None,
                       validation_df=None,
                       many_hot_encoder=None,
                       args=None,
                       exp_name=None,
                       state=None,
                       save_best_eb=None,
                       lr_scheduler=None,
                       warm_start=True):
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
    consistency_criterion = nn.MSELoss().cuda()
    # [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
    #     [class_criterion, consistency_criterion_strong])

    # meters = AverageMeterSet()

    best_iterations = 0
    best_f1 = 0

    # rampup_length = len(strong_loader) * cfg.n_epoch // 2
    global global_step

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = iterations // 2
    avg_strong_loss = 0
    avg_weak_loss = 0

    sample_rate = 44100 if args.n_frames == 864 else 16000
    hop_length = 511 if args.n_frames == 864 else 320

    strong_iter = iter(strong_loader)
    weak_iter = iter(weak_loader)
    unlabel_iter = iter(unlabel_loader)

    strong_iter_ema = iter(strong_loader)
    weak_iter_ema = iter(weak_loader)
    unlabel_iter_ema = iter(unlabel_loader)

    for i in range(1, iterations + 1):
        global_step += 1
        lr_scheduler.step()
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        try:
            strong_sample, strong_target, strong_ids = next(strong_iter)
            strong_sample_ema, strong_target_ema, strong_ids_ema = next(strong_iter_ema)
        except:
            strong_iter = iter(strong_loader)
            strong_sample, strong_target, _ = next(strong_iter)
            strong_iter_ema = iter(strong_loader)
            strong_sample_ema, strong_target_ema, _ = next(strong_iter_ema)
        try:
            weak_sample, weak_target, _ = next(weak_iter)
            weak_sample_ema, weak_target_ema, _ = next(weak_iter_ema)
        except:
            weak_iter = iter(weak_loader)
            weak_sample, weak_target, _ = next(weak_iter)
            weak_iter_ema = iter(weak_loader)
            weak_sample_ema, weak_target_ema, _ = next(weak_iter_ema)
        try:
            unlabel_sample, unlabel_target, _ = next(unlabel_iter)
            unlabel_sample_ema, unlabel_target_ema, _ = next(unlabel_iter_ema)
        except:
            unlabel_iter = iter(unlabel_loader)
            unlabel_sample, _, _ = next(unlabel_iter)
            unlabel_iter_ema = iter(unlabel_loader)
            unlabel_sample_ema, _, _ = next(unlabel_iter_ema)

        # ipdb.set_trace()
        assert strong_ids == strong_ids_ema

        if warm_start and global_step < 2000:

            weak_sample = weak_sample.to('cuda')
            weak_target = weak_target.to('cuda')
            unlabel_sample = unlabel_sample.to('cuda')
            weak_sample_ema = weak_sample_ema.to('cuda')
            weak_target_ema = weak_target_ema.to('cuda')
            unlabel_sample_ema = unlabel_sample_ema.to('cuda')

            pred_strong_ema_w, pred_weak_ema_w = ema_model(weak_sample_ema)
            pred_strong_ema_u, pred_weak_ema_u = ema_model(unlabel_sample_ema)
            pred_strong_ema_w, pred_strong_ema_u = \
                pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
            pred_weak_ema_u, pred_weak_ema_w = \
                pred_weak_ema_w.detach(), pred_weak_ema_u.detach()

            pred_strong_w, pred_weak_w = model(weak_sample)
            pred_strong_u, pred_weak_u = model(unlabel_sample)
            weak_class_loss = class_criterion(pred_weak_w, weak_target)

            # compute consistency loss
            consistency_cost = cfg.max_consistency_cost * rampup_value
            consistency_loss_weak = consistency_cost * consistency_criterion(pred_weak_w, pred_weak_ema_w) \
                                    + consistency_cost * consistency_criterion(pred_weak_u, pred_weak_ema_u)

            logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)

            loss = weak_class_loss + consistency_loss_weak

        else:
            strong_sample, strong_sample_ema = strong_sample.to('cuda'), strong_sample_ema.to('cuda')
            strong_target, strong_target_ema = strong_target.to('cuda'), strong_target_ema.to('cuda')
            weak_sample, weak_sample_ema = weak_sample.to('cuda'), weak_sample_ema.to('cuda')
            weak_target, weak_target_ema = weak_target.to('cuda'), weak_target_ema.to('cuda')
            unlabel_sample, unlabel_sample_ema = unlabel_sample.to('cuda'), unlabel_sample_ema.to('cuda')

            pred_strong_ema_s, pred_weak_ema_s = ema_model(strong_sample_ema)
            pred_strong_ema_w, pred_weak_ema_w = ema_model(weak_sample_ema)
            pred_strong_ema_u, pred_weak_ema_u = ema_model(unlabel_sample_ema)
            pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
                pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
            pred_weak_ema_s, pred_weak_ema_u, pred_weak_ema_w = \
                pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()

            pred_strong_s, pred_weak_s = model(strong_sample)
            pred_strong_w, pred_weak_w = model(weak_sample)
            pred_strong_u, pred_weak_u = model(unlabel_sample)
            strong_class_loss = class_criterion(pred_strong_s, strong_target)
            weak_class_loss = class_criterion(pred_weak_w, weak_target)

            # compute consistency loss
            consistency_cost = cfg.max_consistency_cost * rampup_value
            consistency_loss_strong = consistency_cost * consistency_criterion(pred_strong_s, pred_strong_ema_s) \
                                      + consistency_cost * consistency_criterion(pred_strong_w, pred_strong_ema_w) \
                                      + consistency_cost * consistency_criterion(pred_strong_u, pred_strong_ema_u)
            consistency_loss_weak = consistency_cost * consistency_criterion(pred_weak_s, pred_weak_ema_s) \
                                    + consistency_cost * consistency_criterion(pred_weak_w, pred_weak_ema_w) \
                                    + consistency_cost * consistency_criterion(pred_weak_u, pred_weak_ema_u)

            logger.scalar_summary('train_strong_loss', strong_class_loss.item(), global_step)
            logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)

            loss = strong_class_loss + weak_class_loss + consistency_loss_strong + consistency_loss_weak

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema_variables(model, ema_model, 0.999, global_step)

        if i % log_interval == 0:
            model.eval()
            ema_model.eval()
            with torch.no_grad():
                predictions = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'result_{i}.csv'))
                valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                             sample_rate=sample_rate, hop_length=hop_length)

                predictions = get_batch_predictions(ema_model, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'ema_result_{i}.csv'))
                ema_valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                                 sample_rate=sample_rate, hop_length=hop_length)

            state['model']['state_dict'] = model.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer']['state_dict'] = optimizer.state_dict()
            state['iterations'] = i
            state['valid_metric'] = valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'iteration_{i}.pth'))

            state['model']['state_dict'] = ema_model.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            # state['optimizer']['state_dict'] = optimizer.state_dict()
            state['iterations'] = i
            state['valid_metric'] = ema_valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'ema_iteration_{i}.pth'))

            global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

            if save_best_eb.apply(global_valid):
                best_iterations = i
                best_f1 = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                torch.save(state, model_fname)
            model.train()
            ema_model.train()

    return best_iterations, best_f1


def train_all(strong_loader, weak_loader, unlabel_loader, model, ema_model, optimizer, epoch, logger,
              loss_function='BCE',
              consistency_loss_function='MSE'):
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
    else:
        raise NotImplementedError

    if consistency_loss_function == 'MSE':
        consistency_criterion = nn.MSELoss().cuda()

    global_step = 0

    # vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
    start = time.time()
    rampup_length = len(strong_loader) * cfg.n_epoch // 2
    avg_strong_loss = 0
    avg_weak_loss = 0
    avg_vat_loss = 0
    for i, ((s_batch_input, s_target, s_data),
            (w_batch_input, w_target, w_data),
            (u_batch_input, u_target, u_data)) in \
            enumerate(zip(strong_loader, weak_loader, unlabel_loader)):

        global_step = epoch * len(strong_loader) + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        s_batch_input = s_batch_input.to('cuda')
        s_target = s_target.to('cuda')
        w_batch_input = w_batch_input.to('cuda')
        w_target = w_target.to('cuda')
        u_batch_input = u_batch_input.to('cuda')
        u_target = u_target.to('cuda')

        lds = vat_loss(model, u_batch_input)

        s_strong_pred, s_weak_pred = model(s_batch_input)
        w_strong_pred, w_weak_pred = model(w_batch_input)

        s_strong_pred_ema, s_weak_pred_ema = ema_model(s_batch_input)
        w_strong_pred_ema, w_weak_pred_ema = ema_model(w_batch_input)
        s_strong_pred_ema = s_strong_pred_ema.detach()
        w_weak_pred_ema = w_weak_pred_ema.detach()

        strong_class_loss = class_criterion(s_strong_pred, s_target)
        weak_class_loss = class_criterion(w_weak_pred, w_target)
        ema_weak_class_loss = class_criterion(w_weak_pred_ema, w_target)

        # VAT
        # loss = strong_class_loss + weak_class_loss + lds

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_vat_loss += lds.item() / min(len(strong_loader), len(weak_loader))

    logger.scalar_summary('Ave. strong class loss', avg_strong_loss, epoch)
    logger.scalar_summary('Ave. weak class loss', avg_weak_loss, epoch)
    logger.scalar_summary('Ave. vat loss', avg_vat_loss, epoch)

    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    LOG.info(f'\tAve. vat loss: {avg_vat_loss}')


def to_tensor(numpy_array, cuda=True):
    return torch.from_numpy(numpy_array).cuda()


def train_mixmatch(strong_loader, weak_loader, unlabel_loader, model, optimizer, epoch, logger, loss_function='BCE'):
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
    mm_criterion = MixMatchLoss().to('cuda')

    # consistency_criterion_strong = nn.MSELoss()
    # [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
    #     [class_criterion, consistency_criterion_strong])

    avg_strong_loss = 0
    avg_weak_loss = 0

    for i, ((s_batch_input, s_target, _),
            (w_batch_input, w_target, _),
            (u_batch_input, _, _)) in \
            enumerate(zip(strong_loader, weak_loader, unlabel_loader)):
        # X, U, p, q = mixmatch(s_batch_input.numpy(), s_target.numpy(), u_batch_input.numpy(), model)
        # X, U, p, q = to_tensor(X), to_tensor(U), to_tensor(p), to_tensor(q)
        #
        # # import ipdb
        # # ipdb.set_trace()
        # pred_x, _ = model(X)
        # pred_u, _ = model(U)
        # strong_class_loss = mm_criterion(pred_x, pred_u, p, q)
        s_batch_input = s_batch_input.to('cuda')
        s_target = s_target.to('cuda')

        s_strong_pred, s_weak_pred = model(s_batch_input)
        strong_class_loss = class_criterion(s_strong_pred, s_target)

        X, U, p, q = mixmatch(w_batch_input.numpy(), w_target.numpy(), u_batch_input.numpy(), model, strong=False)
        X, U, p, q = to_tensor(X), to_tensor(U), to_tensor(p).float(), to_tensor(q).float()
        _, pred_x = model(X)
        _, pred_u = model(U)
        # import ipdb
        # ipdb.set_trace()
        weak_class_loss = mm_criterion(pred_x, pred_u, p, q)

        loss = strong_class_loss + weak_class_loss
        #
        # import ipdb
        # ipdb.set_trace()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))

    logger.scalar_summary('Ave. strong class loss', avg_strong_loss, epoch)
    logger.scalar_summary('Ave. weak class loss', avg_weak_loss, epoch)
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')


def train_minimaxDA(strong_loader, weak_loader, unlabel_loader, model_f, model_c, optimizer_f, optimizer_c, epoch,
                    logger, loss_function='BCE', T=0.05, lam=0.1):
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

    avg_strong_loss = 0
    avg_weak_loss = 0

    for i, ((s_batch_input, s_target, _),
            (w_batch_input, w_target, _),
            (u_batch_input, _, _)) in \
            enumerate(zip(strong_loader, weak_loader, unlabel_loader)):
        s_batch_input, s_target = s_batch_input.cuda(), s_target.cuda()
        w_batch_input, w_target = w_batch_input.cuda(), w_target.cuda()
        u_batch_input = u_batch_input.cuda()

        # for source and target
        f_s = model_f(s_batch_input)
        f_s = f_s / f_s.norm(p=2, dim=1, keepdim=True) / T
        p_strong_s, p_weak_s = model_c(f_s)

        f_t = model_f(w_batch_input)
        f_t = f_t / f_t.norm(p=2, dim=1, keepdim=True) / T
        p_strong_t, p_weak_t = model_c(f_t)

        # ipdb.set_trace()

        strong_class_loss = class_criterion(p_strong_s, s_target)
        # weak_class_loss = class_criterion(p_weak_t, w_target) + class_criterion(p_weak_s, s_target.max(-2)[0])
        weak_class_loss = class_criterion(p_weak_t, w_target)

        loss = strong_class_loss + weak_class_loss

        # for unlabel
        f_u = model_f(u_batch_input)
        f_u = f_u / f_u.norm(p=2, dim=1, keepdim=True) / T
        p_strong_u, p_weak_u = model_c(f_u, reverse=True)

        ent = - torch.mean(p_strong_u * torch.log(p_strong_u + 1e-6)) - torch.mean(
                p_weak_u * torch.log(p_weak_u + 1e-6))
        theta_f = loss + lam * ent
        theta_c = loss - lam * ent

        optimizer_f.zero_grad()
        theta_f.backward(retain_graph=True)
        optimizer_f.step()

        optimizer_c.zero_grad()
        theta_c.backward()
        optimizer_c.step()

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))

    logger.scalar_summary('Ave. strong class loss', avg_strong_loss, epoch)
    logger.scalar_summary('Ave. weak class loss', avg_weak_loss, epoch)
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')


def train_unet_strong_weak(strong_loader, weak_loader, model, optimizer, epoch, logger, loss_function='BCE'):
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
    elif loss_function == 'Dice':
        class_criterion = BCEDiceLoss().to('cuda')
    # consistency_criterion_strong = nn.MSELoss()
    # [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
    #     [class_criterion, consistency_criterion_strong])

    # meters = AverageMeterSet()

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(strong_loader) * cfg.n_epoch // 2
    avg_strong_loss = 0
    avg_weak_loss = 0
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
        s_batch_input = s_batch_input.squeeze(1).permute(0, 2, 1)
        w_batch_input = w_batch_input.squeeze(1).permute(0, 2, 1)

        # s_target = s_target.permute(0, 2, 1)
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

        s_pred, _, = model(s_batch_input)
        _, w_pred, = model(w_batch_input)

        # s_pred = s_pred.max(3)[0].permute(0, 2, 1)
        # w_pred, _ = w_pred.max(3)[0].max(2)

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
        # import ipdb
        # ipdb.set_trace()
        # strong_class_loss = class_criterion(s_pred * s_target, s_target)
        strong_class_loss = class_criterion(s_pred, s_target)
        # pdb.set_trace()
        # w_target = w_target.max(-2)[0]
        weak_class_loss = class_criterion(w_pred, w_target)
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

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))
        # global_step += 1
        # if ema_model is not None:
        #     update_ema_variables(model, ema_model, 0.999, global_step)

    # epoch_time = time.time() - start
    logger.scalar_summary('Ave. strong class loss', avg_strong_loss, epoch)
    logger.scalar_summary('Ave. weak class loss', avg_weak_loss, epoch)
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')


# LOG.info(
#     'Epoch: {}\t'
#     'Time {:.2f}\t'
#     '{meters}'.format(
#         epoch, epoch_time, meters=meters))


def get_batch_predictions(model, data_loader, decoder, post_processing=False, save_predictions=None, tta=1,
                          transforms=None, mode='validation', logger=None):
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):

        if tta != 1:
            assert transforms is not None
            mean_strong = None
            mean_weak = None
            for i in range(tta):
                batch_input_np = batch_input.numpy()
                for transform in transforms:
                    for j in range(batch_input.shape[0]):
                        batch_input_np[j] = transform(batch_input_np[j])
                batch_input_t = torch.from_numpy(batch_input_np)
                if torch.cuda.is_available():
                    batch_input_t = batch_input_t.cuda()
                strong, weak = model(batch_input_t)
                if mean_strong is None:
                    mean_strong = strong
                    mean_weak = weak
                else:
                    mean_strong += strong
                    mean_weak += weak
            pred_strong = mean_strong / tta
            pred_weak = mean_weak / tta
        else:
            # strong, weak = model(batch_input)
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            pred_strong, pred_weak = model(batch_input)

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")

    if mode == 'validation' and logger is not None:
        logger.scalar_summary('valid_strong_loss', avg_strong_loss, global_step)
        logger.scalar_summary('valid_weak_loss', avg_weak_loss, global_step)
    return prediction_df


def get_crnn2_batch_predictions(model, data_loader, decoder, post_processing=False, save_predictions=None, tta=1,
                                transforms=None, mode='validation', mask=True):
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    avg_ead_loss = 0
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        pred_strong, pred_weak, pred_ead = model(batch_input)

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            ead_class_loss = class_criterion(pred_ead, target.max(-1)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)
            avg_ead_loss += ead_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)

        if mask:
            pass

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

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")
    # ipdb.set_trace()

    if mode == 'validation':
        LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
        LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
        LOG.info(f'\tAve. weak class loss: {avg_ead_loss}')
    return prediction_df


def train_crnn2_strong_weak(strong_loader, weak_loader, model, optimizer, epoch, logger, loss_function='BCE'):
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
    avg_strong_loss = 0
    avg_weak_loss = 0
    avg_ead_loss = 0
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

        # s_target = s_target.permute(0, 2, 1)
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

        s_pred, _, pred_ead = model(s_batch_input)
        _, w_pred, _ = model(w_batch_input)

        # s_pred = s_pred.max(3)[0].permute(0, 2, 1)
        # w_pred, _ = w_pred.max(3)[0].max(2)

        # ipdb.set_trace()
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
        # import ipdb
        # ipdb.set_trace()
        # strong_class_loss = class_criterion(s_pred * s_target, s_target)
        strong_class_loss = class_criterion(s_pred, s_target)

        ead_class_loss = class_criterion(pred_ead.squeeze(-1), s_target.max(-1)[0])
        # pdb.set_trace()
        # w_target = w_target.max(-2)[0]
        weak_class_loss = class_criterion(w_pred, w_target)
        # meters.update('Strong loss', strong_class_loss.item())

        # strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
        # meters.update('Strong EMA loss', strong_ema_class_loss.item())
        # if loss is not None:
        #     loss += strong_class_loss + weak_class_loss
        # else:
        loss = strong_class_loss + weak_class_loss + ead_class_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))

    logger.scalar_summary('Ave. strong class loss', avg_strong_loss, epoch)
    logger.scalar_summary('Ave. weak class loss', avg_weak_loss, epoch)
    logger.scalar_summary('Ave. ead class loss', avg_ead_loss, epoch)
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    LOG.info(f'\tAve. ead class loss: {avg_ead_loss}')


def get_batch_da_predictions(model_f, model_c, data_loader, decoder, post_processing=False, save_predictions=None,
                             tta=1, transforms=None, mode='validation'):
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):

        if tta != 1:
            assert transforms is not None
            mean_strong = None
            mean_weak = None
            for i in range(tta):
                batch_input_np = batch_input.numpy()
                for transform in transforms:
                    for j in range(batch_input.shape[0]):
                        batch_input_np[j] = transform(batch_input_np[j])
                batch_input_t = torch.from_numpy(batch_input_np)
                if torch.cuda.is_available():
                    batch_input_t = batch_input_t.cuda()
                strong, weak = model(batch_input_t)
                if mean_strong is None:
                    mean_strong = strong
                    mean_weak = weak
                else:
                    mean_strong += strong
                    mean_weak += weak
            pred_strong = mean_strong / tta
            pred_weak = mean_weak / tta
        else:
            # strong, weak = model(batch_input)
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            # pred_strong, pred_weak = model(batch_input)
            f = model_f(batch_input)
            f = f / f.norm(p=2, dim=1, keepdim=True) / 0.05
            pred_strong, pred_weak = model_c(f)

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)

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
    # pdb.set_trace()

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")
    # ipdb.set_trace()

    if mode == 'validation':
        LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
        LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    return prediction_df


def get_unet_batch_predictions(model, data_loader, decoder, post_processing=False, save_predictions=None,
                               mode='validation'):
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        batch_input = batch_input.squeeze(1).permute(0, 2, 1)
        pred_strong, pred_weak = model(batch_input)

        ### Dice
        pred_strong = torch.sigmoid(pred_strong)
        pred_weak = torch.sigmoid(pred_weak)

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)
        # pred_strong = pred_strong.max(3)[0].permute(0, 2, 1)
        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)

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

    if mode == 'validation':
        LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
        LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    return prediction_df


def save_args(args, dest_dir, name='config.yml'):
    import yaml
    print(yaml.dump(vars(args)))
    with open(os.path.join(dest_dir, name), 'w') as f:
        f.write(yaml.dump(vars(args)))


def main(args):
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--outdir', type=str, default='../exp/results',
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
    parser.add_argument('--synth-meta', type=str,
                        default='./DCASE2019_task4/dataset/metadata/train/synthetic.csv',
                        help='Metadata of validation data (csv)')
    parser.add_argument('--valid-meta', type=str,
                        default='./DCASE2019_task4/dataset/metadata/validation/validation.csv',
                        help='Metadata of validation data (csv)')
    parser.add_argument('--iterations', type=int,
                        default=1000,
                        help='Metadata of validation data (csv)')
    parser.add_argument('--log-interval', type=int,
                        default=100,
                        help='Metadata of validation data (csv)')
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                        help='Dropout rate for the decoder')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch-size', '-b', default=8, type=int,
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
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adadelta', 'adam', 'adabound'],
                        help='Optimizer')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--final-lr', default=0.1, type=float,
                        help='Final learning rate for adabound')
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
    parser.add_argument('--pooling-time-ratio', default=1, type=int)
    parser.add_argument('--loss-function', default='BCE', type=str,
                        choices=['BCE', 'FocalLoss', 'Dice'],
                        help='Type of loss function')
    parser.add_argument('--noise-reduction', default=False, type=strtobool)
    parser.add_argument('--pooling-operator', default='auto', type=str,
                        choices=['max', 'min', 'softmax', 'auto', 'cap', 'rap', 'attention'])
    parser.add_argument('--lr-scheduler', default='cosine_annealing', type=str)
    parser.add_argument('--T-max', default=10, type=int,
                        help='Maximum number of iteration for lr scheduling')
    parser.add_argument('--eta-min', default=1e-5, type=float,
                        help='Minimum number of learning rate for lr scheduling')
    parser.add_argument('--train-data', default='original', type=str,
                        choices=['original', 'noise_reduction', 'both'],
                        help='training data')
    parser.add_argument('--test-data', default='original', type=str,
                        choices=['original', 'noise_reduction'],
                        help='test data')
    parser.add_argument('--n-frames', default=500, type=int,
                        help='input frame length')
    parser.add_argument('--mels', default=64, type=int,
                        help='Number of feature mel bins')
    parser.add_argument('--log-mels', default=True, type=strtobool,
                        help='Number of feature mel bins')
    # transfer learning related
    # parser.add_argument('--sed-model', default=False, nargs='?',
    #                     help='Pre-trained SED model')
    # parser.add_argument('--mt-model', default=False, nargs='?',
    #                     help='Pre-trained MT model')
    args = parser.parse_args(args)

    # exp_name = os.path.join('exp', datetime.now().strftime("%Y_%m%d_%H%M%S"))
    exp_name = f'exp3/{datetime.now().strftime("%Y_%m%d")}_model-{args.model}_rir-{args.use_specaugment}' \
               f'_sa-{args.use_specaugment}_pp-{args.use_post_processing}_i-{args.iterations}' \
               f'_ptr-{args.pooling_time_ratio}_l-{args.loss_function}_nr-{args.noise_reduction}' \
               f'_po-{args.pooling_operator}_lrs-{args.lr_scheduler}_{args.T_max}_{args.eta_min}' \
               f'_train-{args.train_data}_test-{args.test_data}_opt-{args.opt}-{args.lr}_mels{args.mels}' \
               f'_logmel{args.log_mels}'
    os.makedirs(os.path.join(exp_name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'log'), exist_ok=True)
    save_args(args, exp_name)
    logger = Logger(exp_name.replace('exp', 'tensorboard'))

    # # read json data
    # if args.use_rir_augmentation:
    #     train_json = './data/train_aug/data_synthetic.json'
    #     train_weak_json = './data/train_aug/data_weak.json'
    #     valid_json = './data/validation/data_validation.json'
    # else:
    #     if args.noise_reduction:
    #         train_json = './data/train_nr/data_synthetic.json'
    #         train_weak_json = './data/train_nr/data_weak.json'
    #         valid_json = './data/validation/data_validation.json'
    #     else:
    #         train_json = './data/train/data_synthetic.json'
    #         train_weak_json = './data/train/data_weak.json'
    #         train_unlabel_json = './data/train/data_unlabel_in_domain.json'
    #         valid_json = './data/validation/data_validation.json'

    sr = '_16k' if args.n_frames == 500 else '_44k'
    mels = '_mel64' if args.mels == 64 else '_mel128'

    train_synth_json = f'./data/train{sr}{mels}/data_synthetic.json'
    train_weak_json = f'./data/train{sr}{mels}/data_weak.json'
    train_unlabel_json = f'./data/train{sr}{mels}/data_unlabel_in_domain.json'
    valid_json = f'./data/validation{sr}{mels}/data_validation.json'

    train_nr_synth_json = f'./data/train{sr}_nr{mels}/data_synthetic.json'
    train_nr_weak_json = f'./data/train{sr}_nr{mels}/data_weak.json'
    train_nr_unlabel_json = f'./data/train{sr}_nr{mels}/data_unlabel_in_domain.json'
    valid_nr_json = f'./data/validation{sr}_nr{mels}/data_validation.json'

    train_nr_synth_json = f'./data/train{sr}{mels}/data_synthetic.json'
    train_nr_weak_json = f'./data/train{sr}{mels}/data_weak.json'
    train_nr_unlabel_json = f'./data/train{sr}{mels}/data_unlabel_in_domain.json'
    valid_nr_json = f'./data/validation{sr}{mels}/data_validation.json'

    synth_df = pd.read_csv(args.synth_meta, header=0, sep="\t")
    validation_df = pd.read_csv(args.valid_meta, header=0, sep="\t")

    with open(train_synth_json, 'rb') as train_synth_json, \
            open(train_weak_json, 'rb') as train_weak_json, \
            open(train_unlabel_json, 'rb') as train_unlabel_json, \
            open(valid_json, 'rb') as valid_json, \
            open(train_nr_synth_json, 'rb') as train_nr_synth_json, \
            open(train_nr_weak_json, 'rb') as train_nr_weak_json, \
            open(train_nr_unlabel_json, 'rb') as train_nr_unlabel_json, \
            open(valid_nr_json, 'rb') as valid_nr_json:

        train_synth_json = json.load(train_synth_json)['utts']
        train_weak_json = json.load(train_weak_json)['utts']
        train_unlabel_json = json.load(train_unlabel_json)['utts']
        valid_json = json.load(valid_json)['utts']

        train_nr_synth_json = json.load(train_nr_synth_json)['utts']
        train_nr_weak_json = json.load(train_nr_weak_json)['utts']
        train_nr_unlabel_json = json.load(train_nr_unlabel_json)['utts']
        valid_nr_json = json.load(valid_nr_json)['utts']

    # transform functions for data loader
    if args.use_specaugment:
        # train_transforms = [Normalize(), TimeWarp(), FrequencyMask(), TimeMask()]
        if args.log_mels:
            train_transforms = [ApplyLog(), Normalize(), FrequencyMask()]
            test_transforms = [ApplyLog(), Normalize(), FrequencyMask()]
        else:
            train_transforms = [Normalize(), FrequencyMask()]
            test_transforms = [Normalize(), FrequencyMask()]
    else:
        if args.log_mels:
            train_transforms = [ApplyLog(), Normalize()]
            test_transforms = [ApplyLog(), Normalize()]
        else:
            train_transforms = [Normalize()]
            test_transforms = [Normalize()]

    if args.train_data == 'original':
        train_synth_dataset = SEDDataset(train_synth_json,
                                         label_type='strong',
                                         sequence_length=args.n_frames,
                                         transforms=train_transforms,
                                         pooling_time_ratio=args.pooling_time_ratio)
        train_weak_dataset = SEDDataset(train_weak_json,
                                        label_type='weak',
                                        sequence_length=args.n_frames,
                                        transforms=train_transforms,
                                        pooling_time_ratio=args.pooling_time_ratio)
        train_unlabel_dataset = SEDDataset(train_unlabel_json,
                                           label_type='unlabel',
                                           sequence_length=args.n_frames,
                                           transforms=train_transforms,
                                           pooling_time_ratio=args.pooling_time_ratio)
    elif args.train_data == 'noise_reduction':
        train_synth_dataset = SEDDataset(train_nr_synth_json,
                                         label_type='strong',
                                         sequence_length=args.n_frames,
                                         transforms=train_transforms,
                                         pooling_time_ratio=args.pooling_time_ratio)
        train_weak_dataset = SEDDataset(train_nr_weak_json,
                                        label_type='weak',
                                        sequence_length=args.n_frames,
                                        transforms=train_transforms,
                                        pooling_time_ratio=args.pooling_time_ratio)
        train_unlabel_dataset = SEDDataset(train_nr_unlabel_json,
                                           label_type='unlabel',
                                           sequence_length=args.n_frames,
                                           transforms=train_transforms,
                                           pooling_time_ratio=args.pooling_time_ratio)
    elif args.train_data == 'both':
        train_org_synth_dataset = SEDDataset(train_synth_json,
                                             label_type='strong',
                                             sequence_length=args.n_frames,
                                             transforms=train_transforms,
                                             pooling_time_ratio=args.pooling_time_ratio)
        train_org_weak_dataset = SEDDataset(train_weak_json,
                                            label_type='weak',
                                            sequence_length=args.n_frames,
                                            transforms=train_transforms,
                                            pooling_time_ratio=args.pooling_time_ratio)
        train_org_unlabel_dataset = SEDDataset(train_unlabel_json,
                                               label_type='unlabel',
                                               sequence_length=args.n_frames,
                                               transforms=train_transforms,
                                               pooling_time_ratio=args.pooling_time_ratio)
        train_nr_synth_dataset = SEDDataset(train_nr_synth_json,
                                            label_type='strong',
                                            sequence_length=args.n_frames,
                                            transforms=train_transforms,
                                            pooling_time_ratio=args.pooling_time_ratio)
        train_nr_weak_dataset = SEDDataset(train_nr_weak_json,
                                           label_type='weak',
                                           sequence_length=args.n_frames,
                                           transforms=train_transforms,
                                           pooling_time_ratio=args.pooling_time_ratio)
        train_nr_unlabel_dataset = SEDDataset(train_nr_unlabel_json,
                                              label_type='unlabel',
                                              sequence_length=args.n_frames,
                                              transforms=train_transforms,
                                              pooling_time_ratio=args.pooling_time_ratio)

        train_synth_dataset = ConcatDataset([train_org_synth_dataset, train_nr_synth_dataset])
        train_weak_dataset = ConcatDataset([train_org_weak_dataset, train_nr_weak_dataset])
        train_unlabel_dataset = ConcatDataset([train_org_unlabel_dataset, train_nr_unlabel_dataset])

    if args.test_data == 'original':
        valid_dataset = SEDDataset(valid_json,
                                   label_type='strong',
                                   sequence_length=args.n_frames,
                                   transforms=test_transforms,
                                   pooling_time_ratio=args.pooling_time_ratio)
    elif args.test_data == 'noise_reduction':
        valid_dataset = SEDDataset(valid_nr_json,
                                   label_type='strong',
                                   sequence_length=args.n_frames,
                                   transforms=test_transforms,
                                   pooling_time_ratio=args.pooling_time_ratio)

    train_synth_loader = DataLoader(train_synth_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_unlabel_loader = DataLoader(train_unlabel_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=args.n_frames // args.pooling_time_ratio)
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
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # For fast training
    torch.backends.cudnn.benchmark = True

    # build model
    crnn_kwargs = cfg.crnn_kwargs
    if args.pooling_time_ratio == 1:
        # crnn_kwargs['pooling'] = list(3 * ((1, 4),))
        if args.mels == 128:
            crnn_kwargs['pooling'] = [(1, 4), (1, 4), (1, 8)]
            # crnn_kwargs["n_RNN_cell"]: 64,
        elif args.mels == 64:
            crnn_kwargs['pooling'] = [(1, 4), (1, 4), (1, 4)]
    elif args.pooling_time_ratio == 8:
        if args.mels == 128:
            crnn_kwargs['pooling'] = [(2, 4), (2, 4), (2, 8)]
    else:
        raise ValueError

    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    # crnn_ema.apply(weights_init)
    crnn = crnn.to('cuda')
    # crnn_ema = crnn_ema.to('cuda')
    #
    # for param in crnn_ema.parameters():
    #     param.detach_()

    sample_rate = 44100 if args.n_frames == 864 else 16000
    hop_length = 511 if args.n_frames == 864 else 320

    # summary(crnn, (1, 864, 64))
    # pdb.set_trace()
    # crnn_ema = CRNN(**crnn_kwargs)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999)}
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    elif args.opt == 'adabound':
        optimizer = adabound.AdaBound(filter(lambda p: p.requires_grad, crnn.parameters()),
                                      lr=args.lr, final_lr=args.final_lr)

    # scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999)}
    # if args.model == 'unet':
    #     net = UNet1D(n_channels=args.mels, n_classes=10).to('cuda')
    #     # summary(net, (64, 864))
    #     if args.opt == 'adam':
    #         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), **optim_kwargs)
    #     elif args.opt == 'adabound':
    #         optimizer = adabound.AdaBound(filter(lambda p: p.requires_grad, net.parameters()),
    #                                       lr=args.lr, final_lr=args.final_lr)
    #     save_best_eb = SaveBest("sup")
    #     save_best_sb = SaveBest("sup")
    #     best_event_epoch = 0
    #     best_event_f1 = 0
    #     best_segment_epoch = 0
    #     best_segment_f1 = 0
    #
    #     sample_rate = 44100 if args.n_frames == 864 else 16000
    #     hop_length = 511 if args.n_frames == 864 else 320
    #
    #     for epoch in range(args.epochs):
    #         net.train()
    #         # TODO: post-process, save best
    #         # MCD.test(validation_df, valid_loader, many_hot_encoder, i)
    #         train_unet_strong_weak(train_synth_loader, train_weak_loader, net, optimizer, epoch, logger,
    #                                args.loss_function)
    #
    #         net.eval()
    #         # if epoch > 50:
    #
    #         with torch.no_grad():
    #             predictions = get_unet_batch_predictions(net, valid_loader, many_hot_encoder.decode_strong)
    #             valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
    #                                                          sample_rate=sample_rate, hop_length=hop_length)
    #             valid_segments_metric = segment_based_evaluation_df(validation_df, predictions,
    #                                                                 time_resolution=float(args.pooling_time_ratio))
    #
    #         global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #         # global_valid = global_valid + np.mean(weak_metric)
    #         if save_best_eb.apply(global_valid):
    #             best_event_epoch = epoch + 1
    #             best_event_f1 = global_valid
    #             model_fname = os.path.join(exp_name, 'model', "best.pth")
    #             # torch.save(state, model_fname)
    #
    #         # For debug
    #         segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #         if save_best_sb.apply(segment_valid):
    #             best_segment_epoch = epoch + 1
    #             best_segment_f1 = segment_valid
    #             model_fname = os.path.join(exp_name, 'model', "best.pth")
    #             # torch.save(state, model_fname)
    #
    #     if cfg.save_best:
    #         model_fname = os.path.join(exp_name, 'model', "best.pth")
    #         # state = torch.load(model_fname)
    #         LOG.info("testing model: {}".format(model_fname))
    #
    #     LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
    #     LOG.info("Event-based: best macro-f1 score: {}".format(best_event_f1))
    #     LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
    #     LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_f1))
    #
    # elif args.model == 'mixmatch':
    #     model = crnn
    #     for epoch in range(args.epochs):
    #         model = model.train()
    #         train_mixmatch(train_synth_loader, train_weak_loader, train_unlabel_loader, model, optimizer, epoch, logger,
    #                        args.loss_function)
    #         # scheduler.step()
    #         model = model.eval()
    #         # if epoch > 50:
    #
    #         with torch.no_grad():
    #             predictions = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
    #                                                 post_processing=args.use_post_processing,
    #                                                 save_predictions=os.path.join(exp_name, 'predictions',
    #                                                                               f'result_{epoch}.csv'))
    #             valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio)
    #
    # if args.model == 'mcd':
    #     # For MCD
    #     G_kwargs = {
    #         "n_in_channel": 1,
    #         "activation"  : "glu",
    #         "dropout"     : 0.5,
    #         "kernel_size" : 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 64, 64],
    #         "pooling"     : list(3 * ((2, 4),))
    #     }
    #     if args.pooling_time_ratio == 1:
    #         G_kwargs['pooling'] = list(3 * ((1, 4),))
    #     F_kwargs = {
    #         "n_class"     : 10, "attention": True, "n_RNN_cell": 64,
    #         "n_layers_RNN": 2,
    #         "dropout"     : 0.5
    #     }
    #     G = Generator(**G_kwargs)
    #     F1 = Classifier(**F_kwargs)
    #     F2 = Classifier(**F_kwargs)
    #
    #     G.apply(weights_init)
    #     F1.apply(weights_init)
    #     F2.apply(weights_init)
    #
    #     optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), **optim_kwargs)
    #     optimizer_f = torch.optim.Adam(filter(lambda p: p.requires_grad, list(F1.parameters()) + list(F2.parameters())),
    #                                    **optim_kwargs)
    #
    #     MCD = MCDSolver(exp_name=exp_name,
    #                     source_loader=train_synth_loader,
    #                     target_loader=train_weak_loader,
    #                     generator=G,
    #                     classifier1=F1,
    #                     classifier2=F2,
    #                     optimizer_g=optimizer_g,
    #                     optimizer_f=optimizer_f,
    #                     num_k=4,
    #                     num_multiply_d_loss=1)
    #
    #     state = {
    #         'generator'         : {"name"      : G.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : G_kwargs,
    #                                'state_dict': G.state_dict()},
    #         'classifier1'       : {"name"      : F1.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : F_kwargs,
    #                                'state_dict': F1.state_dict()},
    #         'classifier2'       : {"name"      : F2.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : F_kwargs,
    #                                'state_dict': F2.state_dict()},
    #         'optimizer_g'       : {"name"      : optimizer_g.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : optim_kwargs,
    #                                'state_dict': optimizer_g.state_dict()},
    #         'optimizer_f'       : {"name"      : optimizer_f.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : optim_kwargs,
    #                                'state_dict': optimizer_f.state_dict()},
    #         "pooling_time_ratio": args.pooling_time_ratio,
    #         'scaler'            : None,
    #         "many_hot_encoder"  : many_hot_encoder.state_dict()
    #     }
    #     save_best_eb = SaveBest("sup")
    #     save_best_sb = SaveBest("sup")
    #     best_event_epoch = 0
    #     best_event_f1 = 0
    #     best_segment_epoch = 0
    #     best_segment_f1 = 0
    #
    #     for epoch in range(args.epochs):
    #         MCD.train()
    #         # TODO: post-process, save best
    #         # MCD.test(validation_df, valid_loader, many_hot_encoder, i)
    #
    #         MCD.set_eval()
    #         # if epoch > 50:
    #
    #         with torch.no_grad():
    #             predictions = MCD.get_batch_predictions(valid_loader, many_hot_encoder, epoch)
    #             valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio)
    #             valid_segments_metric = segment_based_evaluation_df(validation_df, predictions,
    #                                                                 time_resolution=float(args.pooling_time_ratio))
    #             # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
    #         state['generator']['state_dict'] = MCD.G.state_dict()
    #         # state['model_ema']['state_dict'] = crnn_ema.state_dict()
    #         state['optimizer_g']['state_dict'] = MCD.optimizer_g.state_dict()
    #         state['classifier1']['state_dict'] = MCD.F1.state_dict()
    #         # state['model_ema']['state_dict'] = crnn_ema.state_dict()
    #         state['optimizer_f']['state_dict'] = MCD.optimizer_f.state_dict()
    #         state['classifier2']['state_dict'] = MCD.F2.state_dict()
    #         # state['model_ema']['state_dict'] = crnn_ema.state_dict()
    #         state['optimizer_f']['state_dict'] = MCD.optimizer_f.state_dict()
    #         state['epoch'] = epoch
    #         state['valid_metric'] = valid_events_metric.results()
    #         torch.save(state, os.path.join(exp_name, 'model', f'epoch_{epoch + 1}.pth'))
    #
    #         # pdb.set_trace()
    #         global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #         # global_valid = global_valid + np.mean(weak_metric)
    #         if save_best_eb.apply(global_valid):
    #             best_event_epoch = epoch + 1
    #             best_event_f1 = global_valid
    #             model_fname = os.path.join(exp_name, 'model', "best.pth")
    #             torch.save(state, model_fname)
    #
    #         # For debug
    #         segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #         if save_best_sb.apply(segment_valid):
    #             best_segment_epoch = epoch + 1
    #             best_segment_f1 = segment_valid
    #             model_fname = os.path.join(exp_name, 'model', "best.pth")
    #             torch.save(state, model_fname)
    #
    #     if cfg.save_best:
    #         model_fname = os.path.join(exp_name, 'model', "best.pth")
    #         state = torch.load(model_fname)
    #         LOG.info("testing model: {}".format(model_fname))
    #
    #     LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
    #     LOG.info("Event-based: best macro-f1 score: {}".format(best_event_f1))
    #     LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
    #     LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_f1))
    #
    #     # pdb.set_trace()
    #
    # if args.model == 'minimax':
    #     # For MCD
    #     G_kwargs = {
    #         "n_in_channel": 1,
    #         "activation"  : "glu",
    #         "dropout"     : 0.5,
    #         "kernel_size" : 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 64, 64],
    #         "pooling"     : list(3 * ((1, 4),))
    #     }
    #     if args.pooling_time_ratio == 1:
    #         G_kwargs['pooling'] = list(3 * ((1, 4),))
    #     F_kwargs = {
    #         "n_class"     : 10, "attention": True, "n_RNN_cell": 64,
    #         "n_layers_RNN": 2,
    #         "dropout"     : 0.5
    #     }
    #     G = Generator(**G_kwargs).cuda()
    #     F1 = Classifier(**F_kwargs).cuda()
    #     # F2 = Classifier(**F_kwargs)
    #
    #     G.apply(weights_init)
    #     F1.apply(weights_init)
    #     # F2.apply(weights_init)
    #
    #     optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), **optim_kwargs)
    #     optimizer_f = torch.optim.Adam(filter(lambda p: p.requires_grad, F1.parameters()), **optim_kwargs)
    #
    #     state = {
    #         'generator'         : {"name"      : G.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : G_kwargs,
    #                                'state_dict': G.state_dict()},
    #         'classifier1'       : {"name"      : F1.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : F_kwargs,
    #                                'state_dict': F1.state_dict()},
    #         'optimizer_g'       : {"name"      : optimizer_g.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : optim_kwargs,
    #                                'state_dict': optimizer_g.state_dict()},
    #         'optimizer_f'       : {"name"      : optimizer_f.__class__.__name__,
    #                                'args'      : '',
    #                                "kwargs"    : optim_kwargs,
    #                                'state_dict': optimizer_f.state_dict()},
    #         "pooling_time_ratio": args.pooling_time_ratio,
    #         'scaler'            : None,
    #         "many_hot_encoder"  : many_hot_encoder.state_dict()
    #     }
    #     save_best_eb = SaveBest("sup")
    #     save_best_sb = SaveBest("sup")
    #     best_event_epoch = 0
    #     best_event_f1 = 0
    #     best_segment_epoch = 0
    #     best_segment_f1 = 0
    #
    #     for epoch in range(args.epochs):
    #         G = G.train()
    #         F1 = F1.train()
    #         train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=True)
    #         train_unlabel_loader = DataLoader(train_unlabel_dataset, batch_size=args.batch_size, shuffle=True)
    #         train_minimaxDA(train_synth_loader, train_weak_loader, train_unlabel_loader, G, F1, optimizer_g,
    #                         optimizer_f, epoch, logger)
    #         # TODO: post-process, save best
    #         # MCD.test(validation_df, valid_loader, many_hot_encoder, i)
    #
    #         G = G.eval()
    #         F1 = F1.eval()
    #         # if epoch > 50:
    #
    #         with torch.no_grad():
    #             predictions = get_batch_da_predictions(G, F1, valid_loader, many_hot_encoder.decode_strong,
    #                                                    post_processing=args.use_post_processing,
    #                                                    save_predictions=os.path.join(exp_name, 'predictions',
    #                                                                                  f'result_{epoch}.csv'))
    #             valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio)
    #             model_fname = os.path.join(exp_name, 'model', f"{epoch}.pth")
    #             torch.save(state, model_fname)
    #     pdb.set_trace()

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

    ## SAD validation
    # n_samples = len(train_dataset)
    # train_size = int(n_samples * 0.9)
    # subset1_indices = list(range(0, train_size))
    # subset2_indices = list(range(train_size, n_samples))
    # sad_train_dataset = Subset(train_dataset, subset1_indices)
    # sad_valid_dataset = Subset(train_dataset, subset2_indices)
    # sad_train_loader = DataLoader(sad_train_dataset, batch_size=args.batch_size, shuffle=False)
    # sad_valid_loader = DataLoader(sad_valid_dataset, batch_size=args.batch_size, shuffle=True)

    # model training
    for param in crnn_ema.parameters():
        param.detach_()

    crnn = crnn.train()
    # crnn_ema = crnn_ema.train()
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    if args.epochs == 0:
        logging.info('Use iterations mode, total itarations equals to {:.2f} epochs.'.format(
            args.iterations / len(train_synth_loader)))
        train_one_step(train_synth_loader, train_weak_loader, crnn, optimizer, logger, args.loss_function,
                       iterations=args.iterations,
                       log_interval=args.log_interval,
                       valid_loader=valid_loader,
                       validation_df=validation_df,
                       args=args,
                       many_hot_encoder=many_hot_encoder,
                       exp_name=exp_name,
                       state=state,
                       save_best_eb=save_best_eb)

    # train_one_step_ema(train_synth_loader, train_weak_loader, train_unlabel_loader, crnn, crnn_ema, optimizer, logger, args.loss_function,
    #                iterations=args.iterations,
    #                log_interval=args.log_interval,
    #                valid_loader=valid_loader,
    #                validation_df=validation_df,
    #                args=args,
    #                many_hot_encoder=many_hot_encoder,
    #                exp_name=exp_name,
    #                state=state,
    #                save_best_eb=save_best_eb,
    #                lr_scheduler=scheduler)

    else:
        for epoch in tqdm(range(args.epochs)):
            crnn = crnn.train()
            # train(train_loader, crnn, optimizer, epoch)
            # train_strong_only(sad_train_loader, crnn, optimizer, epoch)

            if args.model == 'vat':
                train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=True)
                train_unlabel_loader = DataLoader(train_unlabel_dataset, batch_size=args.batch_size, shuffle=True)
                train_all(train_synth_loader, train_weak_loader, train_unlabel_loader, crnn, optimizer, epoch, logger,
                          args.loss_function)
            else:
                train_strong_weak(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger, args.loss_function)
            scheduler.step()
            crnn = crnn.eval()
            with torch.no_grad():
                predictions = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'result_{epoch}.csv'))
                valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                             sample_rate=sample_rate, hop_length=hop_length)
                # valid_segments_metric = segment_based_evaluation_df(validation_df, predictions, time_resolution=float(args.pooling_time_ratio))
                # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
            state['model']['state_dict'] = crnn.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer']['state_dict'] = optimizer.state_dict()
            state['epoch'] = epoch
            state['valid_metric'] = valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'epoch_{epoch + 1}.pth'))

            global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            # global_valid = global_valid + np.mean(weak_metric)
            if save_best_eb.apply(global_valid):
                best_event_epoch = epoch + 1
                best_event_f1 = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                torch.save(state, model_fname)

        # For debug
        # segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        # if save_best_sb.apply(segment_valid):
        #     best_segment_epoch = epoch + 1
        #     best_segment_f1 = segment_valid
        #     model_fname = os.path.join(exp_name, 'model', "best.pth")
        #     torch.save(state, model_fname)

    model_fname = os.path.join(exp_name, 'model', "best.pth")
    state = torch.load(model_fname)
    LOG.info("testing model: {}".format(model_fname))

    LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
    LOG.info("Event-based: best macro-f1 score: {}".format(best_event_f1))
    LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
    LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_f1))

    params = torch.load(os.path.join(exp_name, 'model', "best.pth"))
    crnn.load(parameters=params['model']['state_dict'])

    predictions = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
                                        post_processing=args.use_post_processing,
                                        save_predictions=os.path.join(exp_name, 'predictions', f'result_{epoch}.csv'))
    valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                 sample_rate=sample_rate, hop_length=hop_length)
    best_th, best_f1 = search_best_threshold(crnn, valid_loader, validation_df, many_hot_encoder, step=0.1,
                                             sample_rate=sample_rate, hop_length=hop_length)
    best_fs, best_f1 = search_best_median(crnn, valid_loader, validation_df, many_hot_encoder,
                                          spans=list(range(3, 31, 2)), sample_rate=sample_rate, hop_length=hop_length)
    best_ag, best_f1 = search_best_accept_gap(crnn, valid_loader, validation_df, many_hot_encoder,
                                              gaps=list(range(3, 30)), sample_rate=sample_rate, hop_length=hop_length)
    best_rd, best_f1 = search_best_remove_short_duration(crnn, valid_loader, validation_df, many_hot_encoder,
                                                         durations=list(range(3, 30)), sample_rate=sample_rate,
                                                         hop_length=hop_length)

    show_best(crnn, valid_loader, many_hot_encoder.decode_strong,
              params=[best_th, best_fs, best_ag, best_rd],
              sample_rate=sample_rate, hop_length=hop_length)
    print('===================')
    print('best_th', best_th)
    print('best_fs', best_fs)
    print('best_ag', best_ag)
    print('best_rd', best_rd)


if __name__ == '__main__':
    main(sys.argv[1:])
