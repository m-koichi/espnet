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

# baseline modules
import sys

sys.path.append('./DCASE2019_task4/baseline')
from models.RNN import BidirectionalGRU
import config as cfg
from models.CRNN import CRNN
from utils.utils import AverageMeterSet, weights_init, ManyHotEncoder, SaveBest
from evaluation_measures import compute_strong_metrics

import pdb

from dataset import SEDDataset
from transforms import Normalize, ApplyLog, GaussianNoise, TimeWarp, FrequencyMask, TimeMask
from solver.mcd import MCDSolver
# from solver.CNN import CNN
# from solver.RNN import RNN

from logger import Logger

from torch.utils.data import DataLoader, ConcatDataset
import torch.backends.cudnn as cudnn

from scipy.signal import medfilt
import torch
import pandas as pd

from datetime import datetime

from dcase_util.data import DecisionEncoder
from dcase_util.data import ProbabilityEncoder

from distutils.util import strtobool

from solver.transformer import Transformer, TransformerSolver

from model_tuning import search_best_threshold, search_best_median, search_best_accept_gap, \
    search_best_remove_short_duration, show_best


import mlflow

def train(solver, validation_loader, validation_df, decoder, args, exp_name, iteration=1000, log_interval=100,
          save_best_eb=None):
    for i in range(1, iteration + 1):
        # ipdb.set_trace()

        best_f1 = 0
        best_iterations = 0
        solver.train_one_step_ema(warm_start=args.warm_start)
        if i % log_interval == 0 and i >= args.transformer_warmup_steps:
            sample_rate = 44100 if args.n_frames == 864 else 16000
            hop_length = 511 if args.n_frames == 864 else 320
            # solver.eval()
            predictions = solver.get_predictions(validation_loader, decoder,
                                                 save_predictions=os.path.join(exp_name, 'predictions',
                                                                               f'iteration_{i}.csv'))
            valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                         sample_rate=sample_rate, hop_length=hop_length)
            # valid_segments_metric = segment_based_evaluation_df(validation_df, predictions,
            #                                                     time_resolution=float(args.pooling_time_ratio))
            solver.save(os.path.join(exp_name, 'model', f'iteration_{i}.pth'),
                        os.path.join(exp_name, 'model', f'ema_iteration_{i}.pth'))

            global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

            if save_best_eb.apply(global_valid):
                best_iterations = i
                best_f1 = global_valid
                solver.save(os.path.join(exp_name, 'model', f'best_iteration.pth'),
                            os.path.join(exp_name, 'model', f'best_ema_iteration.pth'))
            # pdb.set_trace()
            #                             global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            #                             # global_valid = global_valid + np.mean(weak_metric)
            #                             if save_best_eb.apply(global_valid):
            #                                 best_event_epoch = epoch + 1
            #                                 best_event_f1 = global_valid
            #                                 solver.save(os.path.join(args.exp_name, 'model', "best.pth"))


def to_tensor(numpy_array, cuda=True):
    return torch.from_numpy(numpy_array).cuda()


def save_args(args, dest_dir, file_name='config.yml'):
    import yaml
    print(yaml.dump(vars(args)))
    with open(os.path.join(dest_dir, file_name), 'w') as f:
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
    # network architecture
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
    parser.add_argument('--opt', default='noam', type=str,
                        choices=['adadelta', 'adam', 'noam'],
                        help='Optimizer')
    # parser.add_argument('--lr', default=1e-3, type=float,
    #                     help='Learning rate')
    # parser.add_argument('--final-lr', default=0.1, type=float,
    #                     help='Final learning rate for adabound')
    parser.add_argument('--iterations', default=10000, type=int,
                        help='Maximum number of training iterations')
    parser.add_argument('--log-interval', default=100, type=int,
                        help='Interval to check log')
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
    parser.add_argument('--add-noise', default=False, type=strtobool)
    parser.add_argument('--model', default='crnn_baseline_feature', type=str)
    parser.add_argument('--pooling-time-ratio', default=1, type=int)
    parser.add_argument('--loss-function', default='BCE', type=str,
                        choices=['BCE', 'FocalLoss', 'Dice'],
                        help='Type of loss function')
    parser.add_argument('--noise-reduction', default=False, type=strtobool)
    parser.add_argument('--pooling-operator', default='attention', type=str,
                        choices=['max', 'mean', 'softmax', 'auto', 'cap', 'rap', 'attention'])
    parser.add_argument('--train-data', default='original', type=str,
                        choices=['original', 'noise_reduction', 'both'],
                        help='training data')
    parser.add_argument('--test-data', default='original', type=str,
                        choices=['original', 'noise_reduction'],
                        help='test data')
    parser.add_argument('--n-frames', default=500, type=int,
                        help='input frame length')
    parser.add_argument('--mels', default=128, type=int,
                        help='Number of feature mel bins')
    parser.add_argument('--log-mels', default=True, type=strtobool,
                        help='Number of feature mel bins')
    parser.add_argument('--warm-start', default=True, type=strtobool)

    parser.add_argument("--transformer-init", type=str, default="pytorch",
                        choices=["pytorch", "xavier_uniform", "xavier_normal",
                                 "kaiming_uniform", "kaiming_normal"],
                        help='how to initialize transformer parameters')
    parser.add_argument("--transformer-input-layer", type=str, default="linear",
                        choices=["conv2d", "linear", "embed"],
                        help='transformer input layer type')
    parser.add_argument('--transformer-attn-dropout-rate', default=0.5, type=float,
                        help='dropout in transformer attention. use --dropout-rate if None is set')
    parser.add_argument('--transformer-lr', default=10.0, type=float,
                        help='Initial value of learning rate')
    parser.add_argument('--transformer-warmup-steps', default=25000, type=int,
                        help='optimizer warmup steps')
    parser.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                        help='normalize loss by length')
    parser.add_argument('--input-layer-type', default=1, type=int,
                        help='normalize loss by length')
    parser.add_argument('--adim', default=256, type=int)
    parser.add_argument('--aheads', default=4, type=int)
    parser.add_argument('--elayers', default=6, type=int)
    parser.add_argument('--eunits', default=1024, type=int)
    parser.add_argument('--accum-grad', default=2, type=int)

    args = parser.parse_args(args)

    if args.input_layer_type == 1:
        args.transformer_input_layer = 'linear'
    elif args.input_layer_type == 2:
        args.transformer_input_layer = 'conv2d'
    elif args.input_layer_type == 3:
        args.transformer_input_layer = 'linear'
        args.batch_size //= 2
    else:
        raise ValueError

    exp_name = f'exp2/{datetime.now().strftime("%Y_%m%d")}_{args.model}_rir{args.use_rir_augmentation}_fm{args.use_specaugment}' \
               f'_pp{args.use_post_processing}_an{args.add_noise}_iter{args.iterations}' \
               f'_ptr{args.pooling_time_ratio}_loss{args.loss_function}' \
               f'_po-{args.pooling_operator}' \
               f'_train-{args.train_data}_test-{args.test_data}_opt-{args.opt}_mels{args.mels}' \
               f'_logmel{args.log_mels}_tinit-{args.transformer_init}_tinput-{args.transformer_input_layer}' \
               f'_tdo{args.transformer_attn_dropout_rate}_tlr{args.transformer_lr}_twu{args.transformer_warmup_steps}' \
               f'_adim{args.adim}_aheads{args.aheads}_elayers{args.elayers}_eunits{args.eunits}_ag{args.accum_grad}' \
               f'_ilt{args.input_layer_type}'

    os.makedirs(os.path.join(exp_name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'log'), exist_ok=True)

    # logger = Logger(os.path.join(exp_name, 'log'))

    sr = '_16k' if args.n_frames == 500 else '_44k'
    mels = '_mel64' if args.mels == 64 else '_mel128'
    rir = '_rir' if args.use_rir_augmentation else ''

    train_synth_json = f'./data/train{rir}{sr}{mels}/data_synthetic.json'
    train_weak_json = f'./data/train{rir}{sr}{mels}/data_weak.json'
    train_unlabel_json = f'./data/train{rir}{sr}{mels}/data_unlabel_in_domain.json'
    valid_json = f'./data/validation{sr}{mels}/data_validation.json'

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

    train_transforms = []
    test_transforms = []
    if args.log_mels:
        train_transforms.append(ApplyLog())
        test_transforms.append(ApplyLog())
    train_transforms.append(Normalize())
    test_transforms.append(Normalize())
    if args.add_noise:
        train_transforms.append(GaussianNoise())
        test_transforms.append(GaussianNoise())
    if args.use_specaugment:
        train_transforms.append(FrequencyMask())
        test_transforms.append(FrequencyMask())

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

    if args.ngpu > 1:
        args.batch_size *= args.ngpu

    train_synth_loader = DataLoader(train_synth_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                    drop_last=True)
    train_weak_loader = DataLoader(train_weak_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                   drop_last=True)
    train_unlabel_loader = DataLoader(train_unlabel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                      drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

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

    if args.input_layer_type == 1:
        cnn_kwargs = {
            'pooling'   : [(1, 4), (1, 4), (1, 8)],
            'nb_filters': [64, 64, args.mels]
        }
        assert args.transformer_input_layer == 'linear'
        model = Transformer(input_dim=args.mels,
                            n_class=10,
                            args=args,
                            pooling=args.pooling_operator,
                            input_conv=True,
                            cnn_kwargs=cnn_kwargs)
    elif args.input_layer_type == 2:
        assert args.transformer_input_layer == 'conv2d'
        model = Transformer(input_dim=args.mels,
                            n_class=10,
                            args=args,
                            pooling=args.pooling_operator,
                            input_conv=False,
                            cnn_kwargs=None)
    elif args.input_layer_type == 3:
        assert args.transformer_input_layer == 'linear'
        model = Transformer(input_dim=args.mels,
                            n_class=10,
                            args=args,
                            pooling=args.pooling_operator,
                            input_conv=False,
                            cnn_kwargs=None)

    if args.input_layer_type == 1:
        cnn_kwargs = {
            'pooling'   : [(1, 4), (1, 4), (1, 8)],
            'nb_filters': [64, 64, args.mels]
        }
        assert args.transformer_input_layer == 'linear'
        ema_model = Transformer(input_dim=args.mels,
                                n_class=10,
                                args=args,
                                pooling=args.pooling_operator,
                                input_conv=True,
                                cnn_kwargs=cnn_kwargs)
    elif args.input_layer_type == 2:
        assert args.transformer_input_layer == 'conv2d'
        ema_model = Transformer(input_dim=args.mels,
                                n_class=10,
                                args=args,
                                pooling=args.pooling_operator,
                                input_conv=False,
                                cnn_kwargs=None)
    elif args.input_layer_type == 3:
        assert args.transformer_input_layer == 'linear'
        ema_model = Transformer(input_dim=args.mels,
                                n_class=10,
                                args=args,
                                pooling=args.pooling_operator,
                                input_conv=False,
                                cnn_kwargs=None)

    save_args(args, exp_name)
    if args.input_layer_type == 1:
        with open(os.path.join(exp_name, 'cnn_kwargs.pkl'), 'wb') as f:
            import pickle
            pickle.dump(cnn_kwargs, f)

    with open(os.path.join(exp_name, 'args.pkl'), 'wb') as f:
        import pickle
        pickle.dump(args, f)

    for param in ema_model.parameters():
        param.detach_()

    print(model)
    solver = TransformerSolver(model,
                               ema_model,
                               train_synth_loader,
                               train_weak_loader,
                               train_unlabel_loader,
                               exp_name=exp_name,
                               args=args,
                               criterion=torch.nn.BCELoss().cuda(),
                               consistency_criterion=torch.nn.MSELoss().cuda(),
                               accum_grad=args.accum_grad,
                               rampup_length=args.iterations // 2,
                               optimizer='adam',
                               consistency_cost=2,
                               data_parallel=args.ngpu > 1)

    sample_rate = 44100 if args.n_frames == 864 else 16000
    hop_length = 511 if args.n_frames == 864 else 320

    save_best_eb = SaveBest("sup")

    train(solver, valid_loader, validation_df, many_hot_encoder.decode_strong, args, exp_name,
          iteration=args.iterations,
          log_interval=args.log_interval, save_best_eb=save_best_eb)

    params = torch.load(os.path.join(exp_name, 'model', f"iteration_{args.iterations}.pth"))
    solver.load(parameters=params)

    predictions = TransformerSolver.get_batch_predictions(solver, valid_loader, many_hot_encoder.decode_strong,
                                                          post_processing=args.use_post_processing,
                                                          save_predictions=os.path.join(exp_name, 'predictions',
                                                                                        f'result.csv'))
    valid_events_metric = compute_strong_metrics(predictions, validation_df, args.pooling_time_ratio,
                                                 sample_rate=sample_rate, hop_length=hop_length)
    best_th, best_f1 = search_best_threshold(solver, valid_loader, validation_df, many_hot_encoder, step=0.1)
    best_fs, best_f1 = search_best_median(TransformerSolver, valid_loader, validation_df, many_hot_encoder,
                                          spans=list(range(3, 31, 2)),
                                          sample_rate=sample_rate, hop_length=hop_length)
    best_ag, best_f1 = search_best_accept_gap(solver, valid_loader, validation_df, many_hot_encoder,
                                              gaps=list(range(3, 30)),
                                              sample_rate=sample_rate, hop_length=hop_length)
    best_rd, best_f1 = search_best_remove_short_duration(solver, valid_loader, validation_df, many_hot_encoder,
                                                         durations=list(range(3, 30)),
                                                         sample_rate=sample_rate, hop_length=hop_length)
    # LOG.info("Event-based: best macro-f1 setting: {}".format(best_event_epoch))
    # LOG.info("Event-based: best macro-f1 setting: {}".format(best_event_f1))

    # post_process_fn = [functools.partial(fill_up_gap, accepy_gap=list(best_fs.values())),
    #                    functools.partial(search_up_gap, accepy_gap=list(best_fs.values())),
    #                    functools.partial(fill_up_gap, accepy_gap=list(best_fs.values())),]
    show_best(solver, valid_loader, many_hot_encoder.decode_strong, params=[best_th, best_fs, best_ag, best_rd])
    print('===================')
    print('best_th', best_th)
    print('best_fs', best_fs)
    print('best_ag', best_ag)
    print('best_rd', best_rd)


if __name__ == '__main__':
    main(sys.argv[1:])
