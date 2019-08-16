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
from transforms import Normalize, GaussianNoise, TimeWarp, FrequencyMask, TimeMask, ApplyLog
from solver.mcd import MCDSolver
from solver.unet import UNet1D
from solver.adaptive_pooling import AutoPool, ConvNet
# from solver.CNN import CNN
# from solver.RNN import RNN
from solver.CRNN import CRNN

from logger import Logger
from focal_loss import FocalLoss

from torch.utils.data import DataLoader, Subset
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
import ipdb

from dcase_util.data import DecisionEncoder
from dcase_util.data import ProbabilityEncoder

from distutils.util import strtobool

from sklearn.metrics import accuracy_score, precision_score, recall_score

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
        target = target.max(-1)[0]
        target = target.to('cuda')

        strong_pred, weak_pred = model(batch_input)
        strong_pred = strong_pred.squeeze(-1)
        # ipdb.set_trace()

        # pdb.set_trace()
        loss = None
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


def get_batch_predictions(model, data_loader, decoder, post_processing=False, save_predictions=None):
    prediction_df = pd.DataFrame()
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()

        strong, weak = model(batch_input)
        pred_strong, _ = model(batch_input)
        pred_strong = pred_strong.squeeze(-1)
        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)

        # ipdb.set_trace()
        # if post_processing:
        #     for i in range(pred_strong.shape[0]):
        #         pred_strong[i] = median_filt_1d(pred_strong[i])
        #         pred_strong[i] = fill_up_gap(pred_strong[i])
        #         pred_strong[i] = remove_short_duration(pred_strong[i])

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
    return pred_strong


def evaluate(model, data_loader):
    pre = 0
    rec = 0
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()

        strong, weak = model(batch_input)
        pred_strong, _ = model(batch_input)
        pred_strong = pred_strong.squeeze(-1)
        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)
        # if post_processing:
        #     for i in range(pred_strong.shape[0]):
        #         pred_strong[i] = median_filt_1d(pred_strong[i])
        #         pred_strong[i] = fill_up_gap(pred_strong[i])
        #         pred_strong[i] = remove_short_duration(pred_strong[i])

        target = target.max(-1)[0].numpy()
        pre += precision_score(target.reshape(-1), pred_strong.reshape(-1))
        rec += recall_score(target.reshape(-1), pred_strong.reshape(-1))
        # ipdb.set_trace()

        # for pred, data_id in zip(pred_strong, data_ids):
        #     # pred = post_processing(pred)
        #     pred = decoder(pred)
        #     pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
        #     pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
        #     prediction_df = prediction_df.append(pred)

            # if batch_idx == 0:
            #     LOG.debug("predictions: \n{}".format(pred))
            #     LOG.debug("predictions strong: \n{}".format(pred_strong))
            #     prediction_df = pred.copy()
            # else:S
    # pdb.set_trace()
    print(pre / (batch_idx + 1))
    print(rec / (batch_idx + 1))
    #
    # if save_predictions is not None:
    #     LOG.info("Saving predictions at: {}".format(save_predictions))
    #     prediction_df.to_csv(save_predictions, index=False, sep="\t")
    return pred_strong

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
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
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
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam'],
                        help='Optimizer')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
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
                        choices=['BCE', 'FocalLoss'],
                        help='Type of loss function')
    parser.add_argument('--noise-reduction', default=False, type=strtobool)
    parser.add_argument('--pooling-operator', default='auto', type=str,
                        choices=['max', 'min', 'softmax', 'auto', 'cap', 'rap', 'attention'])
    # transfer learning related
    # parser.add_argument('--sed-model', default=False, nargs='?',
    #                     help='Pre-trained SED model')
    # parser.add_argument('--mt-model', default=False, nargs='?',
    #                     help='Pre-trained MT model')
    args = parser.parse_args(args)

    exp_name = os.path.join('exp', datetime.now().strftime("%Y_%m%d_%H%M%S"))
    exp_name = f'exp/{datetime.now().strftime("%Y_%m%d")}_eadmodel-{args.model}_rir-{args.use_specaugment}' \
               f'_sa-{args.use_specaugment}_pp-{args.use_post_processing}_e-{args.epochs}' \
               f'_ptr-{args.pooling_time_ratio}_l-{args.loss_function}_nr-{args.noise_reduction}' \
               f'_po-{args.pooling_operator}'
    os.makedirs(os.path.join(exp_name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'log'), exist_ok=True)
    save_args(args, 'EAD/model_16k')
    logger = Logger(os.path.join(exp_name, 'log'))


    # read json data
    # if args.use_rir_augmentation:
    #     train_json = './data/train_aug/data_synthetic.json'
    #     valid_json = './data/validation/data_validation.json'
    # else:
    #     if args.noise_reduction:
    #         train_json = './data/train_nr/data_synthetic.json'
    #         valid_json = './data/validation/data_validation.json'
    #     else:
    #         train_json = './data/train/data_synthetic.json'
    #         valid_json = './data/validation/data_validation.json'
    train_json = './data/train_16k_mel64/data_synthetic.json'
    valid_json = './data/validation_16k_mel64/data_validation.json'

    with open(train_json, 'rb') as train_json, \
         open(valid_json, 'rb') as valid_json:
        train_json = json.load(train_json)['utts']
        # train_weak_json = json.load(train_weak_json)['utts']
        valid_json = json.load(valid_json)['utts']

    if args.use_specaugment:
        # train_transforms = [Normalize(), TimeWarp(), FrequencyMask(), TimeMask()]
        train_transforms = [ApplyLog(), Normalize(), FrequencyMask()]
        test_transforms = [ApplyLog(),Normalize()]
    else:
        train_transforms = [ApplyLog(),Normalize()]
        test_transforms = [ApplyLog(),Normalize()]
    train_dataset = SEDDataset(train_json, transforms=train_transforms, sequence_length=500, pooling_time_ratio=args.pooling_time_ratio)
    # train_weak_dataset = SEDDataset(train_weak_json, label_type='weak', transforms=train_transforms, pooling_time_ratio=args.pooling_time_ratio)
    valid_dataset = SEDDataset(valid_json, transforms=test_transforms, sequence_length=500, pooling_time_ratio=args.pooling_time_ratio)

    train_synth_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    # train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    validation_df = pd.read_csv(args.valid_meta, header=0, sep="\t")

    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=500)

    # build model
    crnn_kwargs = cfg.crnn_kwargs
    if args.pooling_time_ratio == 1:
        crnn_kwargs['pooling'] = list(3 * ((1, 4),))
        crnn_kwargs['pooling'] = [(1,4),(1,4),(1,8)]
    elif args.pooling_time_ratio == 8:
        pass
    else:
        raise ValueError
    crnn_kwargs['nclass'] = 1
    crnn = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn = crnn.to('cuda')

    # summary(crnn, (1, 864, 64))
    # pdb.set_trace()
    # crnn_ema = CRNN(**crnn_kwargs)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)

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

    ## SAD validation
    n_samples = len(train_dataset)
    train_size = int(n_samples * 0.9)
    subset1_indices = list(range(0, train_size))
    subset2_indices = list(range(train_size, n_samples))
    sad_train_dataset = Subset(train_dataset, subset1_indices)
    sad_valid_dataset = Subset(train_dataset, subset2_indices)
    sad_train_loader = DataLoader(sad_train_dataset, batch_size=args.batch_size, shuffle=False)
    sad_valid_loader = DataLoader(sad_valid_dataset, batch_size=args.batch_size, shuffle=True)

    # model training
    for epoch in tqdm(range(args.epochs)):
        crnn = crnn.train()
        # train(train_loader, crnn, optimizer, epoch)
        # train_strong_only(train_synth_loader, crnn, optimizer, epoch)
        train(sad_train_loader, crnn, optimizer, epoch, args.loss_function)
        model_fname = os.path.join('EAD', 'model_16k', f"epoch{epoch}.pth")
        torch.save(state, model_fname)

        crnn = crnn.eval()
        # if epoch > 50:

        with torch.no_grad():
            # predictions = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
            #                                     post_processing=args.use_post_processing,
            #                                 save_predictions=os.path.join(exp_name, 'predictions', f'result_{epoch}.csv'))
            evaluate(crnn, sad_valid_loader)
            evaluate(crnn, valid_loader)
            # valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio)
            # valid_segments_metric = segment_based_evaluation_df(validation_df, predictions, time_resolution=float(args.pooling_time_ratio))
            # valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)
        # state['model']['state_dict'] = crnn.state_dict()
        # # state['model_ema']['state_dict'] = crnn_ema.state_dict()
        # state['optimizer']['state_dict'] = optimizer.state_dict()
        # state['epoch'] = epoch
        # state['valid_metric'] = valid_events_metric.results()
        # torch.save(state, os.path.join(exp_name, 'model', f'epoch_{epoch + 1}.pth'))
        #
        # # pdb.set_trace()
        # global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        # # global_valid = global_valid + np.mean(weak_metric)
        # if save_best_eb.apply(global_valid):
        #     best_event_epoch = epoch + 1
        #     best_event_f1 = global_valid
        #     model_fname = os.path.join(exp_name, 'model', "best.pth")
        #     torch.save(state, model_fname)
        #
        # # For debug
        # segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        # if save_best_sb.apply(segment_valid):
        #     best_segment_epoch = epoch + 1
        #     best_segment_f1 = segment_valid
        #     model_fname = os.path.join(exp_name, 'model', "best.pth")
        #     torch.save(state, model_fname)

    # if cfg.save_best:
    #     model_fname = os.path.join(exp_name, 'model', "best.pth")
    #     state = torch.load(model_fname)
    #     LOG.info("testing model: {}".format(model_fname))
    #
    # LOG.info("Event-based: best macro-f1 epoch: {}".format(best_event_epoch))
    # LOG.info("Event-based: best macro-f1 score: {}".format(best_event_f1))
    # LOG.info("Segment-based: best macro-f1 epoch: {}".format(best_segment_epoch))
    # LOG.info("Segment-based: best macro-f1 score: {}".format(best_segment_f1))

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
