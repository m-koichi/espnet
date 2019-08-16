

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
import config as cfg
from models.CRNN import CRNN
from utils.Logger import LOG
from utils.utils import AverageMeterSet, weights_init, ManyHotEncoder, SaveBest
from evaluation_measures import compute_strong_metrics
from utils import ramps

import pdb


from dataset import SEDDataset
from transforms import Normalize, GaussianNoise, FrequencyMask, TimeMask

from torch.utils.data import DataLoader

from scipy.signal import medfilt
import torch
import torch.nn as nn
import time
import pandas as pd
import re

from tqdm import tqdm
from datetime import datetime

from dcase_util.data import DecisionEncoder
from dcase_util.data import ProbabilityEncoder

from distutils.util import strtobool


def get_batch_predictions_mcd(model_g, model_f, data_loader, decoder, save_predictions=None):
    prediction_df = pd.DataFrame()
    for batch_idx, (batch_input, _, data_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()

        feature = model_g(batch_input)
        pred_strong, _ = model_f(feature)
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
    return prediction_df