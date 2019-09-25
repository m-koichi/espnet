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

sys.path.append("./DCASE2019_task4/baseline")
from models.CNN import CNN
from models.RNN import BidirectionalGRU
import config as cfg
from utils.Logger import LOG
from utils.utils import weights_init, ManyHotEncoder, SaveBest
from utils import ramps
from evaluation_measures import compute_strong_metrics, segment_based_evaluation_df
import pdb

from dataset import SEDDataset
from transforms import (
    Normalize,
    ApplyLog,
    GaussianNoise,
    FrequencyMask,
    TimeShift,
    FrequencyShift,
    Gain,
)
from solver.mcd import MCDSolver
from solver.unet import UNet1D, BCEDiceLoss

# from solver.CNN import CNN
# from solver.RNN import RNN
from solver.CRNN import CRNN, CRNN_adaBN

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

from model_tuning import (
    search_best_threshold,
    search_best_median,
    search_best_accept_gap,
    search_best_remove_short_duration,
    show_best,
    median_filt_1d,
)

import mlflow
import tempfile
from tensorboardX import SummaryWriter
from solver.transformer import Transformer, TransformerSolver
from functools import wraps
from radam import RAdam

from my_utils import cycle_iteration, get_sample_rate_and_hop_length, ConfMat
from sklearn.metrics import confusion_matrix
import yaml


CLASSES = {
    "Alarm_bell_ringing": 0,
    "Blender": 1,
    "Cat": 2,
    "Dishes": 3,
    "Dog": 4,
    "Electric_shaver_toothbrush": 5,
    "Frying": 6,
    "Running_water": 7,
    "Speech": 8,
    "Vacuum_cleaner": 9,
}


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


def get_mask(Ac, omega=10, sigma=0.5):
    mask = 1 / (1 + torch.exp(-omega * (Ac - sigma)))
    return mask


def get_masked_image(image, Ac):
    mask = get_mask(Ac).permute(1, 0)
    masked_image = torch.zeros((len(CLASSES), 1, 496, 64)).float().cuda()
    for i in range(len(mask)):
        masked_image[i] = (
            mask[i].repeat_interleave(8) * image.permute(0, 2, 1)
        ).permute(0, 2, 1)
        masked_image[i] = image - masked_image[i]
    return masked_image


def mask_image(image, mask):
    image = image.squeeze(0).permute(1, 0)
    masked_image = image * mask.astype(image)
    masked_image = masked_image.permute(1, 0).unsqueeze(0)
    return masked_image


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
            scaling_factor["mean"] += x.numpy()[0, 0, :, :].mean(axis=0)
            scaling_factor["std"] += x.numpy()[0, 0, :, :].std(axis=0)
        scaling_factor["mean"] /= len(dataset)
        scaling_factor["std"] /= len(dataset)

    with open(save_pickle_path, "wb") as f:
        pickle.dump(scaling_factor, f)

    return scaling_factor


def get_batch_predictions(
    model,
    data_loader,
    decoder,
    post_processing=[functools.partial(median_filt_1d, filt_span=39)],
    save_predictions=None,
    transforms=None,
    mode="validation",
    logger=None,
    pooling_time_ratio=1.0,
    sample_rate=22050,
    hop_length=365,
):
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

        if mode == "validation":
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_weak = pred_weak.cpu().data.numpy()

        pred_strong = ProbabilityEncoder().binarization(
            pred_strong, binarization_type="global_threshold", threshold=0.5
        )
        pred_weak = ProbabilityEncoder().binarization(
            pred_weak, binarization_type="global_threshold", threshold=0.5
        )

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
            pred["filename"] = re.sub("^.*?-", "", data_id + ".wav")
            prediction_df = prediction_df.append(pred)

        for i in range(len(pred_strong)):
            tn, fp, fn, tp = confusion_matrix(
                target_np[i].max(axis=0), pred_weak[i], labels=[0, 1]
            ).ravel()
            tag_measure.add_cf(tn, fp, fn, tp)
            for j in range(len(CLASSES)):
                tn, fp, fn, tp = confusion_matrix(
                    target_np[i][:, j], pred_strong[i][:, j], labels=[0, 1]
                ).ravel()
                frame_measure[j].add_cf(tn, fp, fn, tp)

    # In seconds
    prediction_df.onset = (
        prediction_df.onset * pooling_time_ratio / (sample_rate / hop_length)
    )
    prediction_df.offset = (
        prediction_df.offset * pooling_time_ratio / (sample_rate / hop_length)
    )

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
    if mode == "validation" and logger is not None:
        logger.scalar_summary("valid_strong_loss", avg_strong_loss, global_step)
        logger.scalar_summary("valid_weak_loss", avg_weak_loss, global_step)
        logger.scalar_summary("frame_level_macro_f1", macro_f1, global_step)
        logger.scalar_summary("frame_level_ave_precision", ave_precision, global_step)
        logger.scalar_summary("frame_level_ave_recall", ave_recall, global_step)
        logger.scalar_summary("frame_level_weak_f1", weak_f1, global_step)

    elapsed_time = time.time() - start
    print(f"prediction finished. elapsed time: {elapsed_time}")
    print(f"valid_strong_loss: {avg_strong_loss}")
    print(f"valid_weak_loss: {avg_weak_loss}")
    print(f"frame level macro f1: {macro_f1}")
    print(f"frame level ave. precision: {ave_precision}")
    print(f"frame level ave. recall: {ave_recall}")
    print(f"weak f1: {weak_f1}")

    return prediction_df, ave_precision, ave_recall, macro_f1, weak_f1


def save_args(args, dest_dir, name="config.yml"):
    import yaml

    print(yaml.dump(vars(args)))
    with open(os.path.join(dest_dir, name), "w") as f:
        f.write(yaml.dump(vars(args)))


def main(args):
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument("--ngpu", default=0, type=int, help="Number of GPUs")
    parser.add_argument(
        "--outdir", type=str, default="../exp/results", help="Output directory"
    )
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--debugdir", type=str, help="Output directory for debugging")
    parser.add_argument(
        "--resume",
        "-r",
        default="",
        nargs="?",
        help="Resume the training from snapshot",
    )
    parser.add_argument(
        "--minibatches",
        "-N",
        type=int,
        default="-1",
        help="Process only N minibatches (for debug)",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        nargs="?",
        help="Tensorboard log dir path",
    )
    # task related
    parser.add_argument(
        "--train-json",
        type=str,
        default="./data/train_aug/data_synthetic.json",
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--train-weak-json",
        type=str,
        default="./data/train_aug/data_weak.json",
        help="Filename of train weak label data (json)",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default="./data/validation/data_validation.json",
        help="Filename of validation label data (json)",
    )
    parser.add_argument(
        "--synth-meta",
        type=str,
        default="./DCASE2019_task4/dataset/metadata/train/synthetic.csv",
        help="Metadata of validation data (csv)",
    )
    parser.add_argument(
        "--valid-meta",
        type=str,
        default="./DCASE2019_task4/dataset/metadata/validation/validation.csv",
        help="Metadata of validation data (csv)",
    )
    # model (parameter) related
    parser.add_argument(
        "--dropout-rate", default=0.0, type=float, help="Dropout rate for the encoder"
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Dropout rate for the encoder"
    )
    parser.add_argument(
        "--dropout-rate-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder",
    )
    # minibatch related
    parser.add_argument("--batch-size", "-b", default=8, type=int, help="Batch size")
    # optimization related
    parser.add_argument(
        "--opt",
        default="adam",
        type=str,
        choices=["adadelta", "adam", "adabound", "radam"],
        help="Optimizer",
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--final-lr", default=0.1, type=float, help="Final learning rate for adabound"
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Epsilon constant for optimizer"
    )
    parser.add_argument(
        "--eps-decay", default=0.01, type=float, help="Decaying ratio of epsilon"
    )
    parser.add_argument(
        "--weight-decay", default=0.0, type=float, help="Weight decay ratio"
    )
    parser.add_argument(
        "--criterion",
        default="acc",
        type=str,
        choices=["loss", "acc"],
        help="Criterion to perform epsilon decay",
    )
    parser.add_argument(
        "--threshold", default=1e-4, type=float, help="Threshold to stop iteration"
    )
    parser.add_argument(
        "--epochs", "-e", default=30, type=int, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--early-stop-criterion",
        default="validation/main/acc",
        type=str,
        nargs="?",
        help="Value to monitor to trigger an early stopping of the training",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs to wait without improvement before stopping the training",
    )
    parser.add_argument(
        "--grad-clip", default=5, type=float, help="Gradient norm threshold to clip"
    )
    parser.add_argument(
        "--num-save-attention",
        default=3,
        type=int,
        help="Number of samples of attention to be saved",
    )
    parser.add_argument("--use-rir-augmentation", default=False, type=strtobool)
    parser.add_argument("--use-specaugment", default=False, type=strtobool)
    parser.add_argument("--use-post-processing", default=False, type=strtobool)
    parser.add_argument("--model", default="crnn_baseline_feature", type=str)
    parser.add_argument("--pooling-time-ratio", default=1, type=int)
    parser.add_argument(
        "--loss-function",
        default="BCE",
        type=str,
        choices=["BCE", "FocalLoss", "Dice"],
        help="Type of loss function",
    )
    parser.add_argument("--noise-reduction", default=False, type=strtobool)
    parser.add_argument(
        "--pooling-operator",
        default="auto",
        type=str,
        choices=["max", "min", "softmax", "auto", "cap", "rap", "attention"],
    )
    parser.add_argument("--lr-scheduler", default="cosine_annealing", type=str)
    parser.add_argument(
        "--T-max",
        default=10,
        type=int,
        help="Maximum number of iteration for lr scheduling",
    )
    parser.add_argument(
        "--eta-min",
        default=1e-5,
        type=float,
        help="Minimum number of learning rate for lr scheduling",
    )
    parser.add_argument(
        "--train-data",
        default="original",
        type=str,
        choices=["original", "noise_reduction", "both"],
        help="training data",
    )
    parser.add_argument(
        "--test-data",
        default="original",
        type=str,
        choices=["original", "noise_reduction"],
        help="test data",
    )
    parser.add_argument("--n-frames", default=496, type=int, help="input frame length")
    parser.add_argument(
        "--mels", default=64, type=int, help="Number of feature mel bins"
    )
    parser.add_argument(
        "--log-mels", default=True, type=strtobool, help="Number of feature mel bins"
    )
    parser.add_argument(
        "--exp-mode", default="SED", type=str, choices=["SED", "AT", "GAIN", "adaBN"]
    )
    parser.add_argument(
        "--run-name", required=True, type=str, help="run name for mlflow"
    )
    parser.add_argument(
        "--input-type",
        default=1,
        type=int,
        choices=[1, 2, 3],
        help="training dataset for AT. 1:weak only, 2:weak and strong, 3: strong only. (default=1)",
    )
    parser.add_argument(
        "--pretrained", default=None, help="begin training from pre-trained weight"
    )
    parser.add_argument(
        "--averaged", default=False, type=strtobool
    )

    args = parser.parse_args(args)
    averaged = args.averaged
    with open(os.path.join('exp3', args.run_name, 'config.yml')) as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config) 
    
    exp_name = f"exp3/{args.run_name}"

    if args.n_frames == 496:
        sr = "_16k"
    elif args.n_frames == 605:
        sr = "_22k"
    elif args.n_frames == 864:
        sr = "_44k"
    else:
        raise ValueError
    mels = "_mel64" if args.mels == 64 else "_mel128"

    train_synth_json = f"./data/train{sr}{mels}/data_synthetic.json"
    train_weak_json = f"./data/train{sr}{mels}/data_weak.json"
    train_unlabel_json = f"./data/train{sr}{mels}/data_unlabel_in_domain.json"
    valid_json = f"./data/validation{sr}{mels}/data_validation.json"

    synth_df = pd.read_csv(args.synth_meta, header=0, sep="\t")
    validation_df = pd.read_csv(args.valid_meta, header=0, sep="\t")

    with open(train_synth_json, "rb") as train_synth_json, open(
        train_weak_json, "rb"
    ) as train_weak_json, open(train_unlabel_json, "rb") as train_unlabel_json, open(
        valid_json, "rb"
    ) as valid_json:

        train_synth_json = json.load(train_synth_json)["utts"]
        train_weak_json = json.load(train_weak_json)["utts"]
        train_unlabel_json = json.load(train_unlabel_json)["utts"]
        valid_json = json.load(valid_json)["utts"]

    # transform functions for data loader
    if os.path.exists(f"sf{sr}{mels}.pickle"):
        with open(f"sf{sr}{mels}.pickle", "rb") as f:
            scaling_factor = pickle.load(f)
    else:
        train_synth_dataset = SEDDataset(
            train_synth_json,
            label_type="strong",
            sequence_length=args.n_frames,
            transforms=[ApplyLog()],
            pooling_time_ratio=args.pooling_time_ratio,
        )
        train_weak_dataset = SEDDataset(
            train_weak_json,
            label_type="weak",
            sequence_length=args.n_frames,
            transforms=[ApplyLog()],
            pooling_time_ratio=args.pooling_time_ratio,
        )
        train_unlabel_dataset = SEDDataset(
            train_unlabel_json,
            label_type="unlabel",
            sequence_length=args.n_frames,
            transforms=[ApplyLog()],
            pooling_time_ratio=args.pooling_time_ratio,
        )
        scaling_factor = get_scaling_factor(
            [train_synth_dataset, train_weak_dataset, train_unlabel_dataset],
            f"sf{sr}{mels}.pickle",
        )
    scaling = Normalize(mean=scaling_factor["mean"], std=scaling_factor["std"])
    if args.use_specaugment:
        # train_transforms = [Normalize(), TimeWarp(), FrequencyMask(), TimeMask()]
        if args.log_mels:
            train_transforms = [ApplyLog(), scaling, FrequencyMask()]
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

    unsupervised_transforms = [TimeShift(), FrequencyShift()]

    train_synth_dataset = SEDDataset(
        train_synth_json,
        label_type="strong",
        sequence_length=args.n_frames,
        transforms=train_transforms,
        pooling_time_ratio=args.pooling_time_ratio,
    )
    train_weak_dataset = SEDDataset(
        train_weak_json,
        label_type="weak",
        sequence_length=args.n_frames,
        transforms=train_transforms,
        pooling_time_ratio=args.pooling_time_ratio,
    )
    train_unlabel_dataset = SEDDataset(
        train_unlabel_json,
        label_type="unlabel",
        sequence_length=args.n_frames,
        transforms=train_transforms,
        pooling_time_ratio=args.pooling_time_ratio,
    )

    train_synth_dataset_ema = SEDDataset(
        train_synth_json,
        label_type="strong",
        sequence_length=args.n_frames,
        transforms=train_transforms_ema,
        pooling_time_ratio=args.pooling_time_ratio,
    )
    train_weak_dataset_ema = SEDDataset(
        train_weak_json,
        label_type="weak",
        sequence_length=args.n_frames,
        transforms=train_transforms_ema,
        pooling_time_ratio=args.pooling_time_ratio,
    )
    train_unlabel_dataset_ema = SEDDataset(
        train_unlabel_json,
        label_type="unlabel",
        sequence_length=args.n_frames,
        transforms=train_transforms_ema,
        pooling_time_ratio=args.pooling_time_ratio,
    )

    if os.path.exists(f"sf{sr}{mels}.pickle"):
        with open(f"sf{sr}{mels}.pickle", "rb") as f:
            scaling_factor = pickle.load(f)
    else:
        scaling_factor = get_scaling_factor(
            [train_synth_dataset, train_weak_dataset, train_unlabel_dataset],
            f"sf{sr}{mels}.pickle",
        )

    valid_dataset = SEDDataset(
        valid_json,
        label_type="strong",
        sequence_length=args.n_frames,
        transforms=test_transforms,
        pooling_time_ratio=args.pooling_time_ratio,
        time_shift=False,
    )

    train_synth_loader = DataLoader(
        train_synth_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    train_weak_loader = DataLoader(
        train_weak_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    train_unlabel_loader = DataLoader(
        train_unlabel_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    train_synth_loader_ema = DataLoader(
        train_synth_dataset_ema,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    train_weak_loader_ema = DataLoader(
        train_weak_dataset_ema,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    train_unlabel_loader_ema = DataLoader(
        train_unlabel_dataset_ema,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    many_hot_encoder = ManyHotEncoder(
        cfg.classes, n_frames=args.n_frames // args.pooling_time_ratio
    )
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # set random seed
    logging.info("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
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
            crnn_kwargs["pooling"] = [(1, 4), (1, 4), (1, 8)]
            # crnn_kwargs["n_RNN_cell"]: 64,
        elif args.mels == 64:
            crnn_kwargs["pooling"] = [(1, 4), (1, 4), (1, 4)]
    elif args.pooling_time_ratio == 8:
        if args.mels == 128:
            crnn_kwargs["pooling"] = [(2, 4), (2, 4), (2, 8)]
    else:
        raise ValueError

    if sr == "_22k":
        crnn_kwargs = {
            "n_in_channel": 1,
            "nclass": 10,
            "attention": True,
            "n_RNN_cell": 128,
            "n_layers_RNN": 2,
            "activation": "glu",
            "dropout": 0.5,
            "kernel_size": 7 * [3],
            "padding": 7 * [1],
            "stride": 7 * [1],
            "nb_filters": [16, 32, 64, 128, 128, 128, 128],
            "pooling": [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)],
        }

    print(crnn_kwargs)
    if args.exp_mode == "adaBN":
        crnn = CRNN_adaBN(**crnn_kwargs)
    else:
        crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)

    if averaged:
        crnn.load_state_dict(torch.load(os.path.join("exp3", args.run_name, "model", "average.pth")))
    else:
        crnn.load_state_dict(torch.load(os.path.join("exp3", args.run_name, "model", "best.pth"))['model']['state_dict'])
    crnn = crnn.to("cuda")
    crnn_ema = crnn_ema.to("cuda")
    for param in crnn_ema.parameters():
        param.detach_()

    sample_rate, hop_length = get_sample_rate_and_hop_length(args)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999), "weight_decay": 0.0001}
    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs
        )
    elif args.opt == "adabound":
        optimizer = adabound.AdaBound(
            filter(lambda p: p.requires_grad, crnn.parameters()),
            lr=args.lr,
            final_lr=args.final_lr,
        )
    elif args.opt == "radam":
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs
        )

    state = {
        "model": {
            "name": crnn.__class__.__name__,
            "args": "",
            "kwargs": crnn_kwargs,
            "state_dict": crnn.state_dict(),
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "args": "",
            "kwargs": optim_kwargs,
            "state_dict": optimizer.state_dict(),
        },
        "pooling_time_ratio": args.pooling_time_ratio,
        "scaler": scaling_factor,
        "many_hot_encoder": many_hot_encoder.state_dict(),
    }
    save_best_eb = SaveBest("sup")
    save_best_sb = SaveBest("sup")
    best_event_epoch = 0
    best_event_f1 = 0
    best_segment_epoch = 0
    best_segment_f1 = 0

    crnn = crnn.eval()
    #     with torch.no_grad():
    # #                     print('============= For Debug, closed test =============')
    # #                     train_predictions = get_batch_predictions(crnn, train_synth_loader, many_hot_encoder.decode_strong,
    # #                                                          save_predictions=None,
    # #                                                          pooling_time_ratio=args.pooling_time_ratio,
    # #                                                          transforms=None, mode='validation', logger=None,
    # #                                                          sample_rate=sample_rate, hop_length=hop_length)
    # #                     train_events_metric = compute_strong_metrics(train_predictions, synth_df, pooling_time_ratio=None,
    # #                                                                  sample_rate=sample_rate, hop_length=hop_length)

    # #                     print('============= For validation, open test =============')
    #         predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
    #                                                             save_predictions=os.path.join(exp_name, 'predictions',
    #                                                                                           f'result.csv'),
    #                                                             transforms=None, mode='validation', logger=None,
    #                                                             pooling_time_ratio=args.pooling_time_ratio, sample_rate=sample_rate, hop_length=hop_length)
    #         valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
    #                                                                      sample_rate=sample_rate, hop_length=hop_length)

    #         global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #         segment_valid = valid_segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #         with open(os.path.join(exp_name, 'log', f'result_epoch{epoch}.txt'), 'w') as f:
    #                             f.write(f"Event-based macro-f1: {global_valid * 100:.4}\n")
    #                             f.write(f"Segment-based macro-f1: {segment_valid * 100:.4}\n")
    #                             f.write(f"Frame-based macro-f1: {macro_f1 * 100:.4}\n")
    #                             f.write(f"Frame-based ave_precision: {ave_precision * 100:.4}\n")
    #                             f.write(f"Frame-based ave_recall: {ave_recall * 100:.4}\n")
    #                             f.write(f"weak-f1: {weak_f1 * 100:.4}\n")
    #                             f.write(str(valid_events_metric))
    #                             f.write(str(valid_segments_metric))

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

    predictions, ave_precision, ave_recall, macro_f1, weak_f1 = get_batch_predictions(
        crnn,
        valid_loader,
        many_hot_encoder.decode_strong,
        save_predictions=os.path.join(exp_name, "predictions", f"saigen.csv"),
        transforms=None,
        mode="validation",
        logger=None,
        pooling_time_ratio=args.pooling_time_ratio,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    valid_events_metric, valid_segments_metric = compute_strong_metrics(
        predictions,
        validation_df,
        pooling_time_ratio=None,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    
    
    # ==== search best post process parameters ====
    best_th, best_f1 = search_best_threshold(
        crnn,
        valid_loader,
        validation_df,
        many_hot_encoder,
        step=0.1,
        pooling_time_ratio=args.pooling_time_ratio,
        sample_rate=sample_rate,
        hop_length=hop_length
    )

    best_fs, best_f1 = search_best_median(
        crnn,
        valid_loader,
        validation_df,
        many_hot_encoder,
        spans=list(range(1, 31, 2)),
        pooling_time_ratio=args.pooling_time_ratio,
        sample_rate=sample_rate,
        hop_length=hop_length,
        best_th=list(best_th.values())
    )
    
    best_ag, best_f1 = search_best_accept_gap(
        crnn,
        valid_loader,
        validation_df,
        many_hot_encoder,
        gaps=list(range(0, 15)),
        pooling_time_ratio=args.pooling_time_ratio,
        sample_rate=sample_rate,
        hop_length=hop_length,
        best_th=list(best_th.values())
    )
    
    best_rd, best_f1 = search_best_remove_short_duration(
        crnn,
        valid_loader,
        validation_df,
        many_hot_encoder,
        durations=list(range(0, 1)),
        pooling_time_ratio=args.pooling_time_ratio,
        sample_rate=sample_rate,
        hop_length=hop_length,
        best_th=list(best_th.values())
    )
    
    show_best(
        crnn,
        valid_loader,
        validation_df,
        many_hot_encoder,
        pp_params=[best_th, best_fs, best_ag, best_rd],
        pooling_time_ratio=args.pooling_time_ratio,
        sample_rate=sample_rate, hop_length=hop_length
    )
    print('===================')
    print('best_th', best_th)
    print('best_fs', best_fs)
    print('best_ag', best_ag)
    print('best_rd', best_rd)


def compute_frame_level_measures(pred_df, target_df):

    pred = 0
    target = 0
    tn, fp, fn, tp = confusion_matrix(pred, target).ravel()

    recall = tp / (tp + fp)
    precision = tp / (tp + fn)
    f1 = 2 * (recall * precision) / (recall + precision)
    return recall, precision, f1


if __name__ == "__main__":
    main(sys.argv[1:])