#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=2
stop_stage=2

datadir=./dcase20_task4/dataset

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train_nodev"
train_dev="train_dev"
eval_set="eval_set"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download and Synthesized Data Generation"
    git clone https://github.com/turpaultn/dcase20_task4.git
    cd dcase20_task4/scripts
    ./1_download_data.sh
    ./2_generate_data_from_jams.sh
    cd ../../
    cd dcase20_task4
    wget https://zenodo.org/record/3588172/files/DESEDpublic_eval.tar.gz -O ./DESEDpublic_eval.tar.gz
    tar -xzvf DESEDpublic_eval.tar.gz
    cd ../
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}

    python3 local/data_prep.py --train true \
                               --valid true \
                               --eval false \
                               --metadata-dir ${datadir}/metadata \
                               --audio-dir ${datadir}/audio \
                               --sox-option '--norm=-3 -r 16000 -c 1 -t wav' \
                               --remove-dc-component true \


    for x in train dev; do
        for f in wav.scp utt2spk labels.txt; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
