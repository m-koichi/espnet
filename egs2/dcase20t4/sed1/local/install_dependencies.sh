#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh

conda install -y pandas h5py scipy
conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge
conda install -y ffmpeg -c conda-forge

pip install dcase_util
pip install sed_eval
pip install --upgrade psds_eval
pip install scaper
pip install --upgrade desed@git+https://github.com/turpaultn/DESED