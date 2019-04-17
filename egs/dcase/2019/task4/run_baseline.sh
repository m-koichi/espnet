#!/usr/bin/env bash


# Copyright 2019 Nagoya University (Koichi Miyazaki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch # pytorch only
stage=3         # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)


set -e
set -u

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Environment setup. Install dependencies
    echo "stage 0: Environment setup"
    conda install -y pandas h5py scipy
    conda install -y torchvision
    conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge
    conda install -y ffmpeg -c conda-forge

    pip install dcase_util
    pip install sed-eval
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    ### Note: It may not work on python3.7
    echo "stage 1: Data Preparation"
    python local/baseline/download_data.py
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    python local/baseline/main.py
fi
