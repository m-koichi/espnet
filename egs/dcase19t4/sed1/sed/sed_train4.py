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
import functools

# from sed_utils import make_batchset, CustomConverter

sys.path.append('./DCASE2019_task4/baseline')
from models.CNN import CNN
from models.RNN import BidirectionalGRU
import config as cfg
from utils.Logger import LOG
from utils.utils import weights_init, ManyHotEncoder, SaveBest
from utils import ramps
from evaluation_measures import compute_strong_metrics, segment_based_evaluation_df
import pdb

from dataset import SEDDataset, SEDDatasetEMA
from transforms import Normalize, ApplyLog, GaussianNoise, FrequencyMask, TimeShift, FrequencyShift, Gain
from solver.mcd import MCDSolver
from solver.unet import UNet1D, BCEDiceLoss
# from solver.CNN import CNN
# from solver.RNN import RNN
from solver.CRNN import CRNN, CRNN_adaBN, SubSpecCRNN

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
    search_best_remove_short_duration, show_best, median_filt_1d

import mlflow
import tempfile
from tensorboardX import SummaryWriter
from solver.transformer import Transformer, TransformerSolver
from functools import wraps
from radam import RAdam

from my_utils import cycle_iteration, get_sample_rate_and_hop_length, ConfMat
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from CB_loss import CBLoss



CLASSES = {
    'Alarm_bell_ringing'        : 0,
    'Blender'                   : 1,
    'Cat'                       : 2,
    'Dishes'                    : 3,
    'Dog'                       : 4,
    'Electric_shaver_toothbrush': 5,
    'Frying'                    : 6,
    'Running_water'             : 7,
    'Speech'                    : 8,
    'Vacuum_cleaner'            : 9
}

weak_samples_list = [192, 125, 164, 177, 208, 97, 165, 322, 522, 162]
strong_samples_list = [40092, 69093, 28950, 23370, 25153, 51504, 34489, 30453, 122494, 53418]

weak_class_weights = np.sum(weak_samples_list) / (len(CLASSES) * np.array(weak_samples_list))
strong_class_weights = np.sum(strong_samples_list) / (len(CLASSES) * np.array(strong_samples_list))



def elapsed_time(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print("%s took %d time to finish" % (f.__name__, elapsed))
        return result
    return wrapper

global_step = 0


def lr_test(model, train_loader, criterion, optimizer, step=0.1):
    model = model.cuda()
    criterion = criterion.cuda()
    for data, target, _ in train_loader:
        pred_strong, pred_weak = model(data.cuda())
        loss = criterion.cuda()
        scheduler.step()
        print(loss.item())


        
def log_scalar(writer, name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)



def train_strong_weak(strong_loader, weak_loader, model, optimizer, epoch, logger, loss_function='BCE', mode='SED'):
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
        strong_class_criterion = nn.BCELoss().to('cuda')
        weak_class_criterion = nn.BCELoss().to('cuda')
#         weak_class_criterion = nn.MultiLabelSoftMarginLoss().cuda()
    elif loss_function == 'FocalLoss':
        strong_class_criterion = FocalLoss(gamma=2).to('cuda')
        weak_class_criterion = nn.BCELoss().to('cuda')
        
    elif loss_function == 'CBLoss':
        strong_class_criterion = CBLoss(samples_per_cls=torch.from_numpy(strong_class_weights),
                                       loss_type='FocalLoss').to('cuda')
        weak_class_criterion = CBLoss(samples_per_cls=torch.from_numpy(weak_class_weights),
                                     ).to('cuda')

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    avg_strong_loss = 0
    avg_weak_loss = 0
    
    start = time.time()

    for i, ((s_batch_input, s_target, s_data), (w_batch_input, w_target, w_data)) in \
            enumerate(zip(strong_loader, weak_loader)):
        s_batch_input, s_target = s_batch_input.to('cuda'), s_target.cuda()
        w_batch_input, w_target = w_batch_input.to('cuda'), w_target.cuda()

        s_strong_pred, s_weak_pred = model(s_batch_input)
        w_strong_pred, w_weak_pred = model(w_batch_input)
        
        strong_class_loss = strong_class_criterion(s_strong_pred, s_target)
        weak_class_loss = weak_class_criterion(w_weak_pred, w_target)
        
        # meters.update('Strong loss', strong_class_loss.item())
        logger.scalar_summary('train_strong_loss', strong_class_loss.item(), epoch)
        logger.scalar_summary('train_weak_loss', weak_class_loss.item(), epoch)

        if mode == 'SED':
            loss = strong_class_loss + weak_class_loss
        elif mode == 'AT':
            loss = weak_class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'

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
    
    elapsed_time = time.time() - start
    print(f'1 epoch finished. Elapsed time: {elapsed_time}')
    
    
def train_strong_weak_gain(strong_loader, weak_loader, model, optimizer, epoch, logger, loss_function='paper', mode='SED', ext_data=False):
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
        strong_class_criterion = nn.BCELoss().to('cuda')
        weak_class_criterion = nn.BCELoss().to('cuda')
    elif loss_function == 'FocalLoss':
        strong_class_criterion = FocalLoss(gamma=2).to('cuda')
        weak_class_criterion = nn.BCELoss().to('cuda')
    elif loss_function == 'paper':
        strong_class_criterion = nn.MSELoss().cuda()
#         weak_class_criterion = nn.MultiLabelSoftMarginLoss().cuda()
        weak_class_criterion = nn.BCELoss().cuda()
    elif loss_function == 'CBLoss':
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
        

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    avg_strong_loss = 0
    avg_weak_loss = 0
    avg_am_loss = 0
    
    start = time.time()

    for i, ((s_batch_input, s_target, s_data), (w_batch_input, w_target, w_data)) in \
            enumerate(zip(strong_loader, weak_loader)):
        s_batch_input, s_target = s_batch_input.to('cuda'), s_target.cuda()
        w_batch_input, w_target = w_batch_input.to('cuda'), w_target.cuda()

        if ext_data:
            s_strong_pred, s_weak_pred = model(s_batch_input)
        w_strong_pred, w_weak_pred = model(w_batch_input)
        
        am_loss = 0
        for bidx, image in enumerate(w_batch_input):
#             logging.getLogger().setLevel(logging.WARNING)
#             ipdb.set_trace()
            masked_sample = get_masked_image(image, w_strong_pred[bidx].detach())
            masked_sample_strong_pred, masked_sample_weak_pred = model(masked_sample)
            for c in range(len(CLASSES)):
                am_loss += masked_sample_weak_pred[c, c] / len(CLASSES)
        
        if ext_data:
            strong_class_loss = strong_class_criterion(s_strong_pred, s_target)
        weak_class_loss = weak_class_criterion(w_weak_pred, w_target)
        
        # meters.update('Strong loss', strong_class_loss.item())
        if ext_data:
            logger.scalar_summary('train_strong_loss', strong_class_loss.item(), epoch)
        logger.scalar_summary('train_weak_loss', weak_class_loss.item(), epoch)
        logger.scalar_summary('train_am_loss', am_loss.item(), epoch)

        if ext_data:
            loss = 10 * strong_class_loss + weak_class_loss + am_loss
        else:
            loss = weak_class_loss + am_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ext_data:
            avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_am_loss += am_loss.item() / min(len(strong_loader), len(weak_loader))

    # epoch_time = time.time() - start
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    LOG.info(f'\tAve. am loss: {avg_am_loss}')
    
    elapsed_time = time.time() - start
    print(f'1 epoch finished. Elapsed time: {elapsed_time}')
    


def get_mask(Ac, omega=10, sigma=0.5):
    mask = 1 / (1 + torch.exp(-omega * (Ac - sigma)))
    return mask


def get_masked_image(image, Ac):
    mask = get_mask(Ac).permute(1, 0)
    masked_image = torch.zeros((len(CLASSES), 1, 496, 64)).float().cuda()
    for i in range(len(mask)):
        masked_image[i] = (mask[i].repeat_interleave(8) * image.permute(0, 2, 1)).permute(0, 2, 1)
        masked_image[i] = image - masked_image[i]
    return masked_image


def mask_image(image, mask):
    image = image.squeeze(0).permute(1, 0)
    masked_image = image * mask.astype(image)
    masked_image = masked_image.permute(1, 0).unsqueeze(0)
    return masked_image
    
    
def train_at_strong_weak(strong_loader, weak_loader, model, optimizer, epoch, logger, input_type=1):
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
    weak_class_criterion = nn.BCELoss().to('cuda')
    
    avg_weak_loss = 0
    
    start = time.time()

    for i, ((s_batch_input, s_target, s_data), (w_batch_input, w_target, w_data)) in \
            enumerate(zip(strong_loader, weak_loader)):
        s_batch_input, s_target = s_batch_input.to('cuda'), s_target.cuda()
        w_batch_input, w_target = w_batch_input.to('cuda'), w_target.cuda()

        
        if input_type == 1:
#         s_strong_pred, s_weak_pred = model(s_batch_input)
#         w_strong_pred, w_weak_pred = model(w_batch_input)
            _, w_weak_pred = model(w_batch_input)
            loss = weak_class_criterion(w_weak_pred, w_target)
            num_iter = len(weak_loader)
        elif input_type == 2:
            _, s_weak_pred = model(s_batch_input)
            _, w_weak_pred = model(w_batch_input)
            loss = weak_class_criterion(w_weak_pred, w_target) + weak_class_criterion(s_weak_pred, s_target.max(dim=1)[0])
            num_iter = min(len(strong_loader), len(weak_loader))
        elif input_type == 3:
            _, s_weak_pred = model(s_batch_input)
            loss = weak_class_criterion(s_weak_pred, s_target.max(dim=1)[0])
            num_iter = len(strong_loader)
        
        logger.scalar_summary('train_loss', loss.item(), epoch)

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_weak_loss += loss.item() / num_iter

    # epoch_time = time.time() - start
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    
    elapsed_time = time.time() - start
    print(f'1 epoch finished. Elapsed time: {elapsed_time}')
    
    
    
def train_strong_weak_adabn(strong_loader, weak_loader, model, optimizer, epoch, logger, loss_function='BCE', mode='SED'):
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
        strong_class_criterion = nn.BCELoss().to('cuda')
        weak_class_criterion = nn.BCELoss().to('cuda')
    elif loss_function == 'FocalLoss':
        strong_class_criterion = FocalLoss(gamma=2).to('cuda')
        weak_class_criterion = nn.BCELoss().to('cuda')

    # LOG.debug("Nb batches: {}".format(len(train_loader)))
    avg_strong_loss = 0
    avg_weak_loss = 0
    
    start = time.time()

    for i, ((s_batch_input, s_target, s_data), (w_batch_input, w_target, w_data)) in \
            enumerate(zip(strong_loader, weak_loader)):
        s_batch_input, s_target = s_batch_input.to('cuda'), s_target.cuda()
        w_batch_input, w_target = w_batch_input.to('cuda'), w_target.cuda()

        s_strong_pred, s_weak_pred = model(s_batch_input, domain='source')
        loss = strong_class_loss = strong_class_criterion(s_strong_pred, s_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        w_strong_pred, w_weak_pred = model(w_batch_input, domain='target')
        loss = weak_class_loss = weak_class_criterion(w_weak_pred, w_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # meters.update('Strong loss', strong_class_loss.item())
        logger.scalar_summary('train_strong_loss', strong_class_loss.item(), epoch)
        logger.scalar_summary('train_weak_loss', weak_class_loss.item(), epoch)

#         if mode == 'SED':
#             loss = strong_class_loss + weak_class_loss
#         elif mode == 'AT':
#             loss = weak_class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'

        # compute gradient and do optimizer step
        
        
#         loss = weak_class_loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

        avg_strong_loss += strong_class_loss.item() / min(len(strong_loader), len(weak_loader))
        avg_weak_loss += weak_class_loss.item() / min(len(strong_loader), len(weak_loader))

    # epoch_time = time.time() - start
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. strong class loss: {avg_strong_loss}')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    
    elapsed_time = time.time() - start
    print(f'1 epoch finished. Elapsed time: {elapsed_time}')
    

def train_at_strong_weak_da(source_loader, target_loader, model, optimizer, epoch, logger, input_type=1):
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
    weak_class_criterion = nn.BCELoss().to('cuda')
    
    avg_weak_loss = 0
    
    start = time.time()

    for i, ((s_batch_input, s_target, s_data), (w_batch_input, w_target, w_data)) in \
            enumerate(zip(strong_loader, weak_loader)):
        s_batch_input, s_target = s_batch_input.to('cuda'), s_target.cuda()
        t_batch_input, t_target = t_batch_input.to('cuda'), w_target.cuda()

        
        if input_type == 1:
#         s_strong_pred, s_weak_pred = model(s_batch_input)
#         w_strong_pred, w_weak_pred = model(w_batch_input)
            _, w_weak_pred = model(w_batch_input)
            loss = weak_class_criterion(w_weak_pred, w_target)
            num_iter = len(weak_loader)
        elif input_type == 2:
            _, s_weak_pred = model(s_batch_input)
            _, w_weak_pred = model(w_batch_input)
            loss = weak_class_criterion(w_weak_pred, w_target) + weak_class_criterion(s_weak_pred, s_target.max(dim=1)[0])
            num_iter = min(len(strong_loader), len(weak_loader))
        elif input_type == 3:
            _, s_weak_pred = model(s_batch_input)
            loss = weak_class_criterion(s_weak_pred, s_target.max(dim=1)[0])
            num_iter = len(strong_loader)
        
        logger.scalar_summary('train_loss', loss.item(), epoch)

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_weak_loss += loss.item() / num_iter

    # epoch_time = time.time() - start
    LOG.info(f'after {epoch} epoch')
    LOG.info(f'\tAve. weak class loss: {avg_weak_loss}')
    
    elapsed_time = time.time() - start
    print(f'1 epoch finished. Elapsed time: {elapsed_time}')



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def cycle_iteration(iterable):
    while True:
        for i in iterable:
            yield i
            
            
def get_sample_rate_and_hop_length(args):
    if args.n_frames == 864:
        sample_rate = 44100
        hop_length = 511
    elif args.n_frames == 605:
        sample_rate = 22050
        hop_length = 365
    elif args.n_frames == 496:
        sample_rate = 16000
        hop_length = 323
    else:
        raise ValueError
    
    return sample_rate, hop_length


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

    sample_rate, hop_length = get_sample_rate_and_hop_length(args)

    strong_iter = cycle_iteration(strong_loader)
    weak_iter = cycle_iteration(weak_loader)

    for i in tqdm(range(1, iterations + 1)):
        strong_sample, strong_target, _ = next(strong_iter)
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

                predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'result_{i}.csv'),
                                                    transforms=None, mode='validation', logger=None,
                                                    pooling_time_ratio=args.pooling_time_ratio,
                                                    sample_rate=sample_rate, hop_length=hop_length)
                valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                             sample_rate=sample_rate, hop_length=hop_length)
                                                             
                # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
            state['model']['state_dict'] = model.state_dict()
            # state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer']['state_dict'] = optimizer.state_dict()
            state['iterations'] = i
            state['valid_metric'] = valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'iteration_{i}.pth'))

            global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

            if save_best_eb.apply(macro_f1):
                best_iterations = i
                best_f1 = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                torch.save(state, model_fname)
            model.train()

    return best_iterations, best_f1


# def train_at_one_step_ema(strong_loader, weak_loader, unlabel_loader,
#                        strong_loader_ema, weak_loader_ema, unlabel_loader_ema,
#                        model, ema_model, optimizer, logger,
#                        loss_function='BCE', iterations=10000,
#                        log_interval=100,
#                        valid_loader=None,
#                        validation_df=None,
#                        many_hot_encoder=None,
#                        args=None,
#                        exp_name=None,
#                        state=None,
#                        save_best_eb=None,
#                        lr_scheduler=None,
#                        warm_start=False):
#     """ One epoch of a Mean Teacher model
#     :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
#     Should return 3 values: teacher input, student input, labels
#     :param model: torch.Module, model to be trained, should return a weak and strong prediction
#     :param optimizer: torch.Module, optimizer used to train the model
#     :param epoch: int, the current epoch of training
#     :param ema_model: torch.Module, student model, should return a weak and strong prediction
#     :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
#     :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
#     """
#     if loss_function == 'BCE':
#         class_criterion = nn.BCELoss().to('cuda')
#     elif loss_function == 'FocalLoss':
#         class_criterion = FocalLoss(gamma=2).to('cuda')
#     consistency_criterion = nn.MSELoss().cuda()
#     # [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
#     #     [class_criterion, consistency_criterion_strong])

#     # meters = AverageMeterSet()

#     best_iterations = 0
#     best_f1 = 0

#     # rampup_length = len(strong_loader) * cfg.n_epoch // 2
#     global global_step

#     # LOG.debug("Nb batches: {}".format(len(train_loader)))
#     start = time.time()
#     rampup_length = iterations // 2
#     avg_strong_loss = 0
#     avg_weak_loss = 0

#     sample_rate, hop_length = get_sample_rate_and_hop_length(args)

#     strong_iter = cycle_iteration(strong_loader)
#     weak_iter = cycle_iteration(weak_loader)
#     unlabel_iter = cycle_iteration(unlabel_loader)

#     strong_iter_ema = cycle_iteration(strong_loader_ema)
#     weak_iter_ema = cycle_iteration(weak_loader_ema)
#     unlabel_iter_ema = cycle_iteration(unlabel_loader_ema)

#     for i in tqdm(range(1, iterations + 1)):
#         global_step += 1
# #         lr_scheduler.step()
#         if global_step < rampup_length:
#             rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
#         else:
#             rampup_value = 1.0
            
#         strong_sample, strong_sample_ema, strong_target, strong_ids = next(strong_iter)
#         weak_sample, weak_sample_ema, weak_target, weak_ids = next(weak_iter)
#         unlabel_sample, unlabel_sample_ema, unlabel_target, unlabel_ids = next(unlabel_iter)
        
# #         strong_sample_ema, strong_target_ema, strong_ids_ema = next(strong_iter_ema)
# #         weak_sample_ema, weak_target_ema, weak_ids_ema = next(weak_iter_ema)
# #         unlabel_sample_ema, unlabel_target_ema, unlabel_ids_ema = next(unlabel_iter_ema)

# #         assert strong_ids == strong_ids_ema
# #         assert weak_ids == weak_ids_ema
# #         assert unlabel_ids == unlabel_ids_ema

#         if warm_start and global_step < 2000:

#             strong_sample, strong_sample_ema = strong_sample.to('cuda'), strong_sample_ema.to('cuda')
#             strong_target, strong_target_ema = strong_target.to('cuda'), strong_target_ema.to('cuda')
#             weak_sample, weak_sample_ema = weak_sample.to('cuda'), weak_sample_ema.to('cuda')
#             weak_target, weak_target_ema = weak_target.to('cuda'), weak_target_ema.to('cuda')
#             unlabel_sample, unlabel_sample_ema = unlabel_sample.to('cuda'), unlabel_sample_ema.to('cuda')

#             pred_strong_ema_s, pred_weak_ema_s = ema_model(strong_sample_ema)
#             pred_strong_ema_w, pred_weak_ema_w = ema_model(weak_sample_ema)
#             pred_strong_ema_u, pred_weak_ema_u = ema_model(unlabel_sample_ema)
#             pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
#                 pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
#             pred_weak_ema_s, pred_weak_ema_w, pred_weak_ema_u = \
#                 pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()
            

#             pred_strong_s, pred_weak_s = model(strong_sample)
#             pred_strong_w, pred_weak_w = model(weak_sample)
#             pred_strong_u, pred_weak_u = model(unlabel_sample)
#             strong_class_loss = class_criterion(pred_strong_s, strong_target)
#             weak_class_loss = class_criterion(pred_weak_w, weak_target)
#             # compute consistency loss
#             consistency_cost = cfg.max_consistency_cost * rampup_value
#             consistency_loss_weak = consistency_cost * consistency_criterion(pred_weak_w, pred_weak_ema_w) \
#                                     + consistency_cost * consistency_criterion(pred_weak_u, pred_weak_ema_u)

#             logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)

#             loss = weak_class_loss + consistency_loss_weak

#         else:
#             strong_sample, strong_sample_ema = strong_sample.to('cuda'), strong_sample_ema.to('cuda')
#             strong_target, strong_target_ema = strong_target.to('cuda'), strong_target_ema.to('cuda')
#             weak_sample, weak_sample_ema = weak_sample.to('cuda'), weak_sample_ema.to('cuda')
#             weak_target, weak_target_ema = weak_target.to('cuda'), weak_target_ema.to('cuda')
#             unlabel_sample, unlabel_sample_ema = unlabel_sample.to('cuda'), unlabel_sample_ema.to('cuda')

#             pred_strong_ema_s, pred_weak_ema_s = ema_model(strong_sample_ema)
#             pred_strong_ema_w, pred_weak_ema_w = ema_model(weak_sample_ema)
#             pred_strong_ema_u, pred_weak_ema_u = ema_model(unlabel_sample_ema)
#             pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
#                 pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
#             pred_weak_ema_s, pred_weak_ema_w, pred_weak_ema_u = \
#                 pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()

#             pred_strong_s, pred_weak_s = model(strong_sample)
#             pred_strong_w, pred_weak_w = model(weak_sample)
#             pred_strong_u, pred_weak_u = model(unlabel_sample)
            
#             strong_class_loss = class_criterion(pred_strong_s, strong_target)
#             strong_class_ema_loss = consistency_criterion(pred_strong_s, pred_strong_ema_s) \
#                                     + consistency_criterion(pred_strong_w, pred_strong_ema_w) \
#                                     + consistency_criterion(pred_strong_u, pred_strong_ema_u)
            
#             weak_class_loss = class_criterion(pred_weak_w, weak_target)
#             weak_class_ema_loss = consistency_criterion(pred_weak_s, pred_weak_ema_s) \
#                                   + consistency_criterion(pred_weak_w, pred_weak_ema_w) \
#                                   + consistency_criterion(pred_weak_u, pred_weak_ema_u)

#             # compute consistency loss
#             consistency_cost = cfg.max_consistency_cost * rampup_value
#             consistency_loss_strong = consistency_cost * strong_class_ema_loss
#             consistency_loss_weak = consistency_cost * weak_class_ema_loss

#             logger.scalar_summary('train_strong_loss', strong_class_loss.item(), global_step)
#             logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)

#             loss = strong_class_loss + weak_class_loss + consistency_loss_strong + consistency_loss_weak

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         update_ema_variables(model, ema_model, 0.999, global_step)

#         if i % log_interval == 0:
#             model.eval()
#             ema_model.eval()
#             with torch.no_grad():
#                 print("========== student model prediction ==========")
#                 predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
#                                                     post_processing=args.use_post_processing,
#                                                     save_predictions=os.path.join(exp_name, 'predictions',
#                                                                                   f'result_{i}.csv'),
#                                                     transforms=None, mode='validation', logger=None,
#                                                     pooling_time_ratio=args.pooling_time_ratio,
#                                                     sample_rate=sample_rate, hop_length=hop_length)
#                 valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
#                                                              sample_rate=sample_rate, hop_length=hop_length)

                
#                 print("========== mean teacher model prediction ==========")
#                 predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(ema_model, valid_loader, many_hot_encoder.decode_strong,
#                                                     post_processing=args.use_post_processing,
#                                                     save_predictions=os.path.join(exp_name, 'predictions',
#                                                                                   f'ema_result_{i}.csv'),
#                                                     transforms=None, mode='validation', logger=None,
#                                                     pooling_time_ratio=args.pooling_time_ratio,
#                                                     sample_rate=sample_rate, hop_length=hop_length)
#                 ema_valid_events_metric, ema_valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
#                                                                  sample_rate=sample_rate, hop_length=hop_length)

#             state['model']['state_dict'] = model.state_dict()
#             # state['model_ema']['state_dict'] = crnn_ema.state_dict()
#             state['optimizer']['state_dict'] = optimizer.state_dict()
#             state['iterations'] = i
#             state['valid_metric'] = valid_events_metric.results()
#             torch.save(state, os.path.join(exp_name, 'model', f'iteration_{i}.pth'))

#             state['model']['state_dict'] = ema_model.state_dict()
#             # state['model_ema']['state_dict'] = crnn_ema.state_dict()
#             # state['optimizer']['state_dict'] = optimizer.state_dict()
#             state['iterations'] = i
#             state['valid_metric'] = ema_valid_events_metric.results()
#             torch.save(state, os.path.join(exp_name, 'model', f'ema_iteration_{i}.pth'))

#             global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

#             if save_best_eb.apply(global_valid):
#                 best_iterations = i
#                 best_f1 = global_valid
#                 model_fname = os.path.join(exp_name, 'model', "best.pth")
#                 torch.save(state, model_fname)
#             model.train()
#             ema_model.train()

#     return best_iterations, best_f1


def train_one_step_ema(strong_loader, weak_loader, unlabel_loader,
                       model, ema_model, optimizer, logger,
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
                       warm_start=False):
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
    avg_strong_loss_ema = 0
    avg_weak_loss_ema = 0

    sample_rate, hop_length = get_sample_rate_and_hop_length(args)

    strong_iter = cycle_iteration(strong_loader)
    weak_iter = cycle_iteration(weak_loader)
    unlabel_iter = cycle_iteration(unlabel_loader)

    for i in tqdm(range(1, iterations + 1)):
        global_step += 1
#         lr_scheduler.step()
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0
            
        strong_sample, strong_sample_ema, strong_target, strong_ids = next(strong_iter)
        weak_sample, weak_sample_ema, weak_target, weak_ids = next(weak_iter)
        unlabel_sample, unlabel_sample_ema, unlabel_target, unlabel_ids = next(unlabel_iter)

#         if warm_start and global_step < 2000:

#             strong_sample, strong_sample_ema = strong_sample.to('cuda'), strong_sample_ema.to('cuda')
#             strong_target, strong_target_ema = strong_target.to('cuda'), strong_target_ema.to('cuda')
#             weak_sample, weak_sample_ema = weak_sample.to('cuda'), weak_sample_ema.to('cuda')
#             weak_target, weak_target_ema = weak_target.to('cuda'), weak_target_ema.to('cuda')
#             unlabel_sample, unlabel_sample_ema = unlabel_sample.to('cuda'), unlabel_sample_ema.to('cuda')

#             pred_strong_ema_s, pred_weak_ema_s = ema_model(strong_sample_ema)
#             pred_strong_ema_w, pred_weak_ema_w = ema_model(weak_sample_ema)
#             pred_strong_ema_u, pred_weak_ema_u = ema_model(unlabel_sample_ema)
#             pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
#                 pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
#             pred_weak_ema_s, pred_weak_ema_u, pred_weak_ema_w = \
#                 pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()
            

#             pred_strong_s, pred_weak_s = model(strong_sample)
#             pred_strong_w, pred_weak_w = model(weak_sample)
#             pred_strong_u, pred_weak_u = model(unlabel_sample)
#             strong_class_loss = class_criterion(pred_strong_s, strong_target)
#             weak_class_loss = class_criterion(pred_weak_w, weak_target)
#             # compute consistency loss
#             consistency_cost = cfg.max_consistency_cost * rampup_value
#             consistency_loss_weak = consistency_cost * consistency_criterion(pred_weak_w, pred_weak_ema_w) \
#                                     + consistency_cost * consistency_criterion(pred_weak_u, pred_weak_ema_u)

#             logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)

#             loss = weak_class_loss + consistency_loss_weak

        strong_sample, strong_sample_ema = strong_sample.to('cuda'), strong_sample_ema.to('cuda')
        strong_target = strong_target.to('cuda')
        weak_sample, weak_sample_ema = weak_sample.to('cuda'), weak_sample_ema.to('cuda')
        weak_target = weak_target.to('cuda')
        unlabel_sample, unlabel_sample_ema = unlabel_sample.to('cuda'), unlabel_sample_ema.to('cuda')

        pred_strong_ema_s, pred_weak_ema_s = ema_model(strong_sample_ema)
        pred_strong_ema_w, pred_weak_ema_w = ema_model(weak_sample_ema)
        pred_strong_ema_u, pred_weak_ema_u = ema_model(unlabel_sample_ema)
        pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
            pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
        pred_weak_ema_s, pred_weak_ema_w, pred_weak_ema_u = \
            pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()

        pred_strong_s, pred_weak_s = model(strong_sample)
        pred_strong_w, pred_weak_w = model(weak_sample)
        pred_strong_u, pred_weak_u = model(unlabel_sample)
        
        strong_class_loss = class_criterion(pred_strong_s, strong_target)
        strong_class_ema_loss = consistency_criterion(pred_strong_s, pred_strong_ema_s) \
                                + consistency_criterion(pred_strong_w, pred_strong_ema_w) \
                                + consistency_criterion(pred_strong_u, pred_strong_ema_u)

        weak_class_loss = class_criterion(pred_weak_w, weak_target)
        weak_class_ema_loss = consistency_criterion(pred_weak_s, pred_weak_ema_s) \
                              + consistency_criterion(pred_weak_w, pred_weak_ema_w) \
                              + consistency_criterion(pred_weak_u, pred_weak_ema_u)

        # compute consistency loss
        consistency_cost = cfg.max_consistency_cost * rampup_value
        consistency_loss_strong = consistency_cost * strong_class_ema_loss
        consistency_loss_weak = consistency_cost * weak_class_ema_loss

        logger.scalar_summary('train_strong_loss', strong_class_loss.item(), global_step)
        logger.scalar_summary('train_weak_loss', weak_class_loss.item(), global_step)
        
        avg_strong_loss += strong_class_loss.item() / log_interval
        avg_weak_loss += weak_class_loss.item() / log_interval
        avg_strong_loss_ema += consistency_loss_strong.item() / log_interval
        avg_weak_loss_ema += consistency_loss_weak.item() / log_interval

        loss = strong_class_loss + weak_class_loss + consistency_loss_strong + consistency_loss_weak

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema_variables(model, ema_model, 0.999, global_step)
        
        if i % log_interval == 0:
            model.eval()
            ema_model.eval()
            with torch.no_grad():
                print("========== student model prediction ==========")
                predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'result_{i}.csv'),
                                                    transforms=None, mode='validation', logger=None,
                                                    pooling_time_ratio=args.pooling_time_ratio,
                                                    sample_rate=sample_rate, hop_length=hop_length)
                valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                             sample_rate=sample_rate, hop_length=hop_length)

                print("========== mean teacher model prediction ==========")
                predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(ema_model, valid_loader, many_hot_encoder.decode_strong,
                                                    post_processing=args.use_post_processing,
                                                    save_predictions=os.path.join(exp_name, 'predictions',
                                                                                  f'ema_result_{i}.csv'),
                                                    transforms=None, mode='validation', logger=None,
                                                    pooling_time_ratio=args.pooling_time_ratio,
                                                    sample_rate=sample_rate, hop_length=hop_length)
                ema_valid_events_metric, ema_valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                                 sample_rate=sample_rate, hop_length=hop_length)

            state['model']['state_dict'] = model.state_dict()
            state['ema_model']['state_dict'] = ema_model.state_dict()
            state['optimizer']['state_dict'] = optimizer.state_dict()
            state['iterations'] = i
            state['valid_metric'] = ema_valid_events_metric.results()
            torch.save(state, os.path.join(exp_name, 'model', f'iteration_{i}.pth'))

#             state['ema_model']['state_dict'] = ema_model.state_dict()
#             # state['model_ema']['state_dict'] = crnn_ema.state_dict()
#             # state['optimizer']['state_dict'] = optimizer.state_dict()
#             state['iterations'] = i
#             state['valid_metric'] = ema_valid_events_metric.results()
#             torch.save(state, os.path.join(exp_name, 'model', f'ema_iteration_{i}.pth'))

            global_valid = ema_valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            segment_valid = ema_valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            with open(os.path.join(exp_name, 'log', f'result_iteration{i}.txt'), 'w') as f:
                f.write(f"Event-based macro-f1: {global_valid * 100:.4}\n")
                f.write(f"Segment-based macro-f1: {segment_valid * 100:.4}\n")
                f.write(f"Frame-based macro-f1: {macro_f1 * 100:.4}\n")
                f.write(f"Frame-based ave_precision: {ave_precision * 100:.4}\n")
                f.write(f"Frame-based ave_recall: {ave_recall * 100:.4}\n")
                f.write(f"weak-f1: {weak_f1 * 100:.4}\n")
                f.write(str(ema_valid_events_metric))
                f.write(str(ema_valid_segments_metric))
            LOG.info(f'after {i} iteration')
            LOG.info(f'\t Ave. strong class loss: {avg_strong_loss}')
            LOG.info(f'\t Ave. weak class loss: {avg_weak_loss}')
            LOG.info(f'\t Ave. consistency loss strong: {avg_strong_loss_ema}')
            LOG.info(f'\t Ave. consistency loss weak: {avg_weak_loss_ema}')
            avg_strong_loss = 0
            avg_weak_loss = 0
            avg_strong_loss_ema = 0
            avg_weak_loss_ema = 0
            

#             if save_best_eb.apply(global_valid):
            if save_best_eb.apply(macro_f1):
                best_iterations = i
                best_f1 = global_valid
                model_fname = os.path.join(exp_name, 'model', "best.pth")
                with open(os.path.join(exp_name, 'log', f'result_iteration{i}.txt'), 'w') as f:
                    f.write(f"Event-based macro-f1: {global_valid * 100:.4}\n")
                    f.write(f"Segment-based macro-f1: {segment_valid * 100:.4}\n")
                    f.write(f"Frame-based macro-f1: {macro_f1 * 100:.4}\n")
                    f.write(f"Frame-based ave_precision: {ave_precision * 100:.4}\n")
                    f.write(f"Frame-based ave_recall: {ave_recall * 100:.4}\n")
                    f.write(f"weak-f1: {weak_f1 * 100:.4}\n")
                    f.write(str(ema_valid_events_metric))
                    f.write(str(ema_valid_segments_metric))
                torch.save(state, model_fname)
            model.train()
            ema_model.train()

    return best_iterations, best_f1


def to_tensor(numpy_array, cuda=True):
    return torch.from_numpy(numpy_array).cuda()


def get_scaling_factor(datasets, save_pickle_path):
    """Get scaling factor on mel spectrogram for each bins
    Args:
        dataset:
        save_path:
    Return:
        mean: (np.ndarray) scaling factor
        std: (np.ndarray) scaling factor
    """
    scaling_factor = {}
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=1)

        for x, _, _ in dataloader:
            if len(scaling_factor) == 0:
                scaling_factor["mean"] = np.zeros(x.size(-1))
                scaling_factor["std"] = np.zeros(x.size(-1))
            scaling_factor["mean"] += x.numpy()[0,0,:,:].mean(axis=0)
            scaling_factor["std"] += x.numpy()[0,0,:,:].std(axis=0)
        scaling_factor["mean"] /= len(dataset)
        scaling_factor["std"] /= len(dataset)

    with open(save_pickle_path, 'wb') as f:
        pickle.dump(scaling_factor, f)

    return scaling_factor



def get_batch_predictions(model, data_loader, decoder, post_processing=[functools.partial(median_filt_1d, filt_span=39)],
                          save_predictions=None,
                          transforms=None, mode='validation', logger=None,
                          pooling_time_ratio=1., sample_rate=22050, hop_length=365):
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    
    # Flame level 
    frame_measure = [ConfMat() for i in range(len(CLASSES))]
    tag_measure = ConfMat()
    
    start = time.time()
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):

        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        pred_strong, pred_weak = model(batch_input)
        
        target_np = target.numpy()

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_weak = pred_weak.cpu().data.numpy()
        
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)
        pred_weak = ProbabilityEncoder().binarization(pred_weak, binarization_type="global_threshold",
                                                        threshold=0.5)
        
        post_processing = None

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            # ipdb.set_trace()
            if post_processing is not None:
                for i in range(pred_strong.shape[0]):
                    for post_process_fn in post_processing:
                        # ipdb.set_trace()
                        pred_strong[i] = post_process_fn(pred_strong[i])
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)
            
        for i in range(len(pred_strong)):
            tn, fp, fn, tp = confusion_matrix(target_np[i].max(axis=0), pred_weak[i], labels=[0,1]).ravel()
            tag_measure.add_cf(tn, fp, fn, tp)
            for j in range(len(CLASSES)):
                tn, fp, fn, tp = confusion_matrix(target_np[i][:, j], pred_strong[i][:, j], labels=[0,1]).ravel()
                frame_measure[j].add_cf(tn, fp, fn, tp)
        

    # In seconds
    prediction_df.onset = prediction_df.onset * pooling_time_ratio / (sample_rate / hop_length)
    prediction_df.offset = prediction_df.offset * pooling_time_ratio / (sample_rate / hop_length)
    
    # Compute frame level macro f1 score
    macro_f1 = 0
    ave_precision = 0
    ave_recall = 0
    for i in range(len(CLASSES)):
        ave_precision_, ave_recall_, macro_f1_ = frame_measure[i].calc_f1()
        ave_precision += ave_precision_
        ave_recall += ave_recall_
        macro_f1 += macro_f1_
    ave_precision /= len(CLASSES)
    ave_recall /= len(CLASSES)
    macro_f1 /= len(CLASSES)
    

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")

    weak_f1 = tag_measure.calc_f1()[2]    
    if mode == 'validation' and logger is not None:
        logger.scalar_summary('valid_strong_loss', avg_strong_loss, global_step)
        logger.scalar_summary('valid_weak_loss', avg_weak_loss, global_step)
        logger.scalar_summary('frame_level_macro_f1', macro_f1, global_step)
        logger.scalar_summary('frame_level_ave_precision', ave_precision, global_step)
        logger.scalar_summary('frame_level_ave_recall', ave_recall, global_step)
        logger.scalar_summary('frame_level_weak_f1', weak_f1, global_step)
        
    elapsed_time = time.time() - start
    print(f'prediction finished. elapsed time: {elapsed_time}')
    print(f'valid_strong_loss: {avg_strong_loss}')
    print(f'valid_weak_loss: {avg_weak_loss}')
    print(f'frame level macro f1: {macro_f1}')
    print(f'frame level ave. precision: {ave_precision}')
    print(f'frame level ave. recall: {ave_recall}')
    print(f'weak f1: {weak_f1}')
    
    return prediction_df, ave_precision, ave_recall, macro_f1, weak_f1



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
    parser.add_argument('--verbose', '-V', default=1, type=int,
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
    parser.add_argument('--batch-size', '-b', default=8, type=int,
                        help='Batch size')
    # optimization related
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adadelta', 'adam', 'adabound', 'radam'],
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
    parser.add_argument('--da_noise', default=False, type=strtobool)
    parser.add_argument('--da_timeshift', default=False, type=strtobool)
    parser.add_argument('--da_freqshift', default=False, type=strtobool)
    
    parser.add_argument('--model', default='crnn_baseline_feature', type=str)
    parser.add_argument('--pooling-time-ratio', default=1, type=int)
    parser.add_argument('--loss-function', default='BCE', type=str,
                        choices=['BCE', 'FocalLoss', 'Dice', 'CBLoss'],
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
    parser.add_argument('--n-frames', default=496, type=int,
                        help='input frame length')
    parser.add_argument('--mels', default=64, type=int,
                        help='Number of feature mel bins')
    parser.add_argument('--log-mels', default=True, type=strtobool,
                        help='Number of feature mel bins')
    parser.add_argument('--exp-mode', default='SED', type=str,
                        choices=['SED', 'AT', 'GAIN', 'adaBN', 'SubSpec'])
    parser.add_argument('--run-name', required=True, type=str,
                        help='run name for mlflow')
    parser.add_argument('--input-type', default=1, type=int, choices=[1, 2, 3],
                        help='training dataset for AT. 1:weak only, 2:weak and strong, 3: strong only. (default=1)')
    parser.add_argument('--pretrained', default=None,
                        help='begin training from pre-trained weight')
    parser.add_argument('--ssl', default=False, type=strtobool,
                        help='semi supervised learning (mean teacher)')

    args = parser.parse_args(args)

    # exp_name = os.path.join('exp', datetime.now().strftime("%Y_%m%d_%H%M%S"))
    os.makedirs(os.path.join('exp3', args.run_name), exist_ok=True)
    exp_name = f'exp3/{args.run_name}/{datetime.now().strftime("%Y_%m%d")}_model-{args.model}_rir-{args.use_specaugment}' \
               f'_sa-{args.use_specaugment}_pp-{args.use_post_processing}_i-{args.iterations}' \
               f'_ptr-{args.pooling_time_ratio}_l-{args.loss_function}_nr-{args.noise_reduction}' \
               f'_po-{args.pooling_operator}_lrs-{args.lr_scheduler}_{args.T_max}_{args.eta_min}' \
               f'_train-{args.train_data}_test-{args.test_data}_opt-{args.opt}-{args.lr}_mels{args.mels}' \
               f'_logmel{args.log_mels}_mode{args.exp_mode}'
    exp_name = f'exp3/{args.run_name}'
    os.makedirs(os.path.join(exp_name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'log'), exist_ok=True)
    save_args(args, exp_name)
    logger = Logger(exp_name.replace('exp', 'tensorboard'))

    if args.n_frames == 496:
        sr = '_16k'
    elif args.n_frames == 605:
        sr = '_22k'
    elif args.n_frames == 864:
        sr = '_44k'
    else:
        raise ValueError
    mels = '_mel64' if args.mels == 64 else '_mel128'
    rir = '_rir' if args.use_rir_augmentation else ''

    train_synth_json = f'./data/train{sr}{mels}{rir}/data_synthetic.json'
    train_weak_json = f'./data/train{sr}{mels}{rir}/data_weak.json'
    train_unlabel_json = f'./data/train{sr}{mels}{rir}/data_unlabel_in_domain.json'
    valid_json = f'./data/validation{sr}{mels}{rir}/data_validation.json'

    synth_df = pd.read_csv(args.synth_meta, header=0, sep="\t")
    validation_df = pd.read_csv(args.valid_meta, header=0, sep="\t")

    with open(train_synth_json, 'rb') as train_synth_json, \
            open(train_weak_json, 'rb') as train_weak_json, \
            open(train_unlabel_json, 'rb') as train_unlabel_json, \
            open(valid_json, 'rb') as valid_json:

        train_synth_json = json.load(train_synth_json)['utts']
        train_weak_json = json.load(train_weak_json)['utts']
        train_unlabel_json = json.load(train_unlabel_json)['utts']
        valid_json = json.load(valid_json)['utts']

    # transform functions for data loader
    if os.path.exists(f"sf{sr}{mels}{rir}.pickle"):
        with open(f"sf{sr}{mels}{rir}.pickle", "rb") as f:
            scaling_factor = pickle.load(f)
    else:
        train_synth_dataset = SEDDataset(train_synth_json,
                                         label_type='strong',
                                         sequence_length=args.n_frames,
                                         transforms=[ApplyLog()],
                                         pooling_time_ratio=args.pooling_time_ratio)
        train_weak_dataset = SEDDataset(train_weak_json,
                                        label_type='weak',
                                        sequence_length=args.n_frames,
                                        transforms=[ApplyLog()],
                                        pooling_time_ratio=args.pooling_time_ratio)
        train_unlabel_dataset = SEDDataset(train_unlabel_json,
                                           label_type='unlabel',
                                           sequence_length=args.n_frames,
                                           transforms=[ApplyLog()],
                                           pooling_time_ratio=args.pooling_time_ratio)
        scaling_factor = get_scaling_factor([train_synth_dataset,
                                            train_weak_dataset,
                                            train_unlabel_dataset],
                                            f"sf{sr}{mels}{rir}.pickle")
    scaling = Normalize(mean=scaling_factor["mean"], std=scaling_factor["std"])
    
    if args.use_specaugment:
        # train_transforms = [Normalize(), TimeWarp(), FrequencyMask(), TimeMask()]
        if args.log_mels:
            train_transforms = [ApplyLog(), scaling, FrequencyMask(), Gain()]
            train_transforms_ema = [ApplyLog(), scaling, GaussianNoise()]
            test_transforms = [ApplyLog(), scaling]
        else:
            train_transforms = [scaling, FrequencyMask()]
            test_transforms = [scaling]
    else:
        if args.log_mels:
            train_transforms = [ApplyLog(), scaling, Gain()]
            train_transforms_ema = [ApplyLog(), scaling, GaussianNoise()]
            test_transforms = [ApplyLog(), scaling]
        else:
            train_transforms = [scaling]
            test_transforms = [scaling]

            
    train_transforms = [ApplyLog(), scaling]
    if args.use_specaugment:
        train_transforms.append(FrequencyMask())
    if args.da_noise:
        train_transforms.append(GaussianNoise())
    if args.da_timeshift:
        train_transforms.append(TimeShift())
    if args.da_freqshift:
        train_transforms.append(FrequencyShift())
    unsupervised_transforms = [TimeShift(), FrequencyShift()]
    train_transforms_ema = [ApplyLog(), scaling, GaussianNoise()]
    test_transforms = [ApplyLog(), scaling]
    
    if args.ssl:
        train_transforms.append(GaussianNoise())
    
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
        
    if args.ssl:
        train_synth_dataset = SEDDatasetEMA(train_synth_json,
                                             label_type='strong',
                                             sequence_length=args.n_frames,
                                             transforms=train_transforms,
                                             pooling_time_ratio=args.pooling_time_ratio)
        train_weak_dataset = SEDDatasetEMA(train_weak_json,
                                            label_type='weak',
                                            sequence_length=args.n_frames,
                                            transforms=train_transforms,
                                            pooling_time_ratio=args.pooling_time_ratio)
        train_unlabel_dataset = SEDDatasetEMA(train_unlabel_json,
                                               label_type='unlabel',
                                               sequence_length=args.n_frames,
                                               transforms=train_transforms,
                                               pooling_time_ratio=args.pooling_time_ratio)

    valid_dataset = SEDDataset(valid_json,
                                   label_type='strong',
                                   sequence_length=args.n_frames,
                                   transforms=test_transforms,
                                   pooling_time_ratio=args.pooling_time_ratio,
                                   time_shift=False)

    train_synth_loader = DataLoader(train_synth_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_unlabel_loader = DataLoader(train_unlabel_dataset, batch_size=args.batch_size*2, shuffle=True, drop_last=False)
    
    
#     train_synth_loader_ema = DataLoader(train_synth_dataset_ema, batch_size=args.batch_size, shuffle=False, drop_last=True)
#     train_weak_loader_ema = DataLoader(train_weak_dataset_ema, batch_size=args.batch_size, shuffle=False, drop_last=True)
#     train_unlabel_loader_ema = DataLoader(train_unlabel_dataset_ema, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
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
    elif args.pooling_time_ratio == 4:
        crnn_kwargs['pooling'] = [(2, 4), (2, 4), (1, 4)]
    elif args.pooling_time_ratio == 8:
        if args.mels == 128:
            crnn_kwargs['pooling'] = [(2, 4), (2, 4), (2, 8)]
    else:
        raise ValueError

    if sr == '_22k':
        crnn_kwargs = {"n_in_channel": 1, "nclass": 10, "attention": True, "n_RNN_cell": 128,
                       "n_layers_RNN": 2,
                       "activation"  : "glu",
                       "dropout"     : 0.5,
                       "kernel_size" : 7 * [3], "padding": 7 * [1], "stride": 7 * [1],
                       "nb_filters"  : [16, 32, 64, 128, 128, 128, 128],
                       "pooling"     : [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]}

    if args.exp_mode == 'SubSpec':
        crnn_kwargs['pooling'] = [(2, 4), (2, 4), (2, 2)]
        crnn = SubSpecCRNN(**crnn_kwargs)
    elif args.exp_mode == 'adaBN':
        crnn = CRNN_adaBN(**crnn_kwargs)
    else:
        crnn = CRNN(**crnn_kwargs)
    print(crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)
    if args.pretrained is not None:
        print(f'load pretrained model: {args.pretrained}')
        parameters = torch.load(os.path.join(exp_name, 'model', 'best.pth'))['model']
        ipdb.set_trace()
        crnn.load(parameters=parameters['state_dict'])
    crnn = crnn.to('cuda')
    crnn_ema = crnn_ema.to('cuda')
    for param in crnn_ema.parameters():
        param.detach_()
    
    sample_rate, hop_length = get_sample_rate_and_hop_length(args)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999), "weight_decay": 0.0001}
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    elif args.opt == 'adabound':
        optimizer = adabound.AdaBound(filter(lambda p: p.requires_grad, crnn.parameters()),
                                      lr=args.lr, final_lr=args.final_lr)
    elif args.opt == 'radam':
        optimizer = RAdam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)


    state = {
        'model'             : {"name"      : crnn.__class__.__name__,
                               'args'      : '',
                               "kwargs"    : crnn_kwargs,
                               'state_dict': crnn.state_dict()},
        'ema_model'         : {"name"      : crnn.__class__.__name__,
                               'args'      : '',
                               "kwargs"    : crnn_kwargs,
                               'state_dict': crnn_ema.state_dict()},
        'optimizer'         : {"name"      : optimizer.__class__.__name__,
                               'args'      : '',
                               "kwargs"    : optim_kwargs,
                               'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": args.pooling_time_ratio,
        'scaler'            : scaling_factor,
        "many_hot_encoder"  : many_hot_encoder.state_dict()
    }
    save_best_eb = SaveBest("sup")
    save_best_sb = SaveBest("sup")
    best_event_epoch = 0
    best_event_f1 = 0
    best_segment_epoch = 0
    best_segment_f1 = 0

    crnn = crnn.train()
    crnn_ema = crnn_ema.train()
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()
        writer = SummaryWriter(output_dir)
        print("Writing TensorBoard events locally to %s\n" % output_dir)

        if args.epochs == 0:
            logging.info('Use iterations mode, total itarations equals to {:.2f} epochs.'.format(
                args.iterations / len(train_synth_loader)))
            
            if args.ssl:
                train_one_step_ema(train_synth_loader, train_weak_loader, train_unlabel_loader,
                                   crnn, crnn_ema, optimizer, logger,
                                   loss_function='BCE',
                                   iterations=args.iterations,
                                   log_interval=args.log_interval,
                                   valid_loader=valid_loader,
                                   validation_df=validation_df,
                                   many_hot_encoder=many_hot_encoder,
                                   args=args,
                                   exp_name=exp_name,
                                   state=state,
                                   save_best_eb=save_best_eb,
                                   lr_scheduler=None,
                                   warm_start=False)
            else:
                train_one_step(train_synth_loader, train_weak_loader, crnn, optimizer, logger, args.loss_function,
                               iterations=args.iterations,
                               log_interval=len(train_synth_loader),
                               valid_loader=valid_loader,
                               validation_df=validation_df,
                               args=args,
                               many_hot_encoder=many_hot_encoder,
                               exp_name=exp_name,
                               state=state,
                               save_best_eb=save_best_eb)

    #         train_one_step_ema(train_synth_loader, train_weak_loader, train_unlabel_loader,
    #                            train_synth_loader_ema, train_weak_loader_ema, train_unlabel_loader_ema,
    #                            crnn, crnn_ema, optimizer, logger,
    #                            loss_function='BCE',
    #                            iterations=args.iterations,
    #                            log_interval=len(train_synth_loader),
    #                            valid_loader=valid_loader,
    #                            validation_df=validation_df,
    #                            many_hot_encoder=many_hot_encoder,
    #                            args=args,
    #                            exp_name=exp_name,
    #                            state=state,
    #                            save_best_eb=save_best_eb,
    #                            lr_scheduler=scheduler,
    #                            warm_start=False)

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
            for epoch in tqdm(range(1, args.epochs+1)):
                global global_step
                global_step = epoch
                global_valid = None
                crnn = crnn.train()
                # train(train_loader, crnn, optimizer, epoch)
                # train_strong_only(sad_train_loader, crnn, optimizer, epoch)
#                 if args.model == 'ema':
#                     train_strong_weak_ema(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger, args.loss_function)
#                 else:
#                     train_strong_weak(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger, args.loss_function,
#                                      mode=args.exp_mode)

                if args.exp_mode == 'SED' or args.exp_mode == 'SubSpec':
                    train_strong_weak(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger, args.loss_function)
                elif args.exp_mode == 'AT':
                    train_at_strong_weak(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger, input_type=args.input_type)

                elif args.exp_mode == 'GAIN':
                    train_strong_weak_gain(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger)
                elif args.exp_mode == 'adaBN':
                    train_strong_weak_adabn(train_synth_loader, train_weak_loader, crnn, optimizer, epoch, logger)
                else:
                    raise NotImplementedError
                    
                scheduler.step()
                crnn = crnn.eval()
                with torch.no_grad():
#                     print('============= For Debug, closed test =============')
#                     train_predictions = get_batch_predictions(crnn, train_synth_loader, many_hot_encoder.decode_strong,
#                                                          save_predictions=None,
#                                                          pooling_time_ratio=args.pooling_time_ratio,
#                                                          transforms=None, mode='validation', logger=None,
#                                                          sample_rate=sample_rate, hop_length=hop_length)
#                     train_events_metric = compute_strong_metrics(train_predictions, synth_df, pooling_time_ratio=None,
#                                                                  sample_rate=sample_rate, hop_length=hop_length)


#                     print('============= For validation, open test =============')
                    if epoch > 20:
                        predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
                                                            save_predictions=os.path.join(exp_name, 'predictions',
                                                                                          f'result_epoch{epoch}.csv'),
                                                            transforms=None, mode='validation', logger=None,
                                                            pooling_time_ratio=args.pooling_time_ratio, sample_rate=sample_rate, hop_length=hop_length)
                        valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                                     sample_rate=sample_rate, hop_length=hop_length)
                
                        
                    
                        global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
                        segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
                        with open(os.path.join(exp_name, 'log', f'result_epoch{epoch}.txt'), 'w') as f:
                            f.write(f"Event-based macro-f1: {global_valid * 100:.4}\n")
                            f.write(f"Segment-based macro-f1: {segment_valid * 100:.4}\n")
                            f.write(f"Frame-based macro-f1: {macro_f1 * 100:.4}\n")
                            f.write(f"Frame-based ave_precision: {ave_precision * 100:.4}\n")
                            f.write(f"Frame-based ave_recall: {ave_recall * 100:.4}\n")
                            f.write(f"weak-f1: {weak_f1 * 100:.4}\n")
                            f.write(str(valid_events_metric))
                            f.write(str(valid_segments_metric))

#                         valid_segments_metric = segment_based_evaluation_df(validation_df, predictions, time_resolution=float(args.pooling_time_ratio))
                    # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)


                if global_valid is not None:
                    state['model']['state_dict'] = crnn.state_dict()
                    # state['model_ema']['state_dict'] = crnn_ema.state_dict()
                    state['optimizer']['state_dict'] = optimizer.state_dict()
                    state['epoch'] = epoch + 1
                    state['valid_metric'] = valid_events_metric.results()
                    torch.save(state, os.path.join(exp_name, 'model', f'epoch_{epoch + 1}.pth'))
#                     if save_best_eb.apply(macro_f1):
                    if save_best_eb.apply(macro_f1):
                        best_event_epoch = epoch + 1
                        best_event_f1 = global_valid
                        model_fname = os.path.join(exp_name, 'model', "best.pth")
                        torch.save(state, model_fname)

#             # For debug
#                 segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
#                 if save_best_sb.apply(segment_valid):
#                     best_segment_epoch = epoch + 1
#                     best_segment_f1 = segment_valid
            #     model_fname = os.path.join(exp_name, 'model', "best.pth")
            #     torch.save(state, model_fname)
        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
            os.path.join(mlflow.get_artifact_uri(), "events"))

#     model_fname = os.path.join(exp_name, 'model', "best.pth")
#     state = torch.load(model_fname)
#     LOG.info("testing model: {}".format(model_fname))

    LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
    LOG.info("Event-based: best macro-f1 score: {}".format(best_event_f1))
    LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
    LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_f1))
    
#     with open(f'log/{exp_name}.txt', 'w') as f:
#         f.write("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
#         f.write("Event-based: best macro-f1 score: {}".format(best_event_f1))
#         f.write("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
#         f.write("Segment-based: best macro-f1 score: {}".format(best_segment_f1))

#     params = torch.load(os.path.join(exp_name, 'model', "best.pth"))
#     crnn.load(parameters=params['model']['state_dict'])

#     predictions = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
#                                         save_predictions=os.path.join(exp_name, 'predictions', f'result_{epoch}.csv'),
#                                         transforms=None, mode='validation', logger=None,
#                                         pooling_time_ratio=args.pooling_time_ratio, sample_rate=sample_rate, hop_length=hop_length)
#     valid_events_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=args.pooling_time_ratio,
#                                                  sample_rate=sample_rate, hop_length=hop_length)
#     best_th, best_f1 = search_best_threshold(crnn, valid_loader, validation_df, many_hot_encoder, step=0.1,
#                                              sample_rate=sample_rate, hop_length=hop_length)
#     best_fs, best_f1 = search_best_median(crnn, valid_loader, validation_df, many_hot_encoder,
#                                           spans=list(range(3, 31, 2)), sample_rate=sample_rate, hop_length=hop_length)
#     best_ag, best_f1 = search_best_accept_gap(crnn, valid_loader, validation_df, many_hot_encoder,
#                                               gaps=list(range(3, 30)), sample_rate=sample_rate, hop_length=hop_length)
#     best_rd, best_f1 = search_best_remove_short_duration(crnn, valid_loader, validation_df, many_hot_encoder,
#                                                          durations=list(range(3, 30)), sample_rate=sample_rate,
#                                                          hop_length=hop_length)

#     show_best(crnn, valid_loader, many_hot_encoder.decode_strong,
#               params=[best_th, best_fs, best_ag, best_rd],
#               sample_rate=sample_rate, hop_length=hop_length)
#     print('===================')
#     print('best_th', best_th)
#     print('best_fs', best_fs)
#     print('best_ag', best_ag)
#     print('best_rd', best_rd)


def compute_frame_level_measures(pred_df, target_df):
    
    pred = 0
    target = 0 
    tn, fp, fn, tp = confusion_matrix(pred, target).ravel()
    
    recall = tp / (tp + fp)
    precision = tp / (tp + fn)
    f1 = 2 * (recall * precision) / (recall + precision)
    return recall, precision, f1

if __name__ == '__main__':
    main(sys.argv[1:])
