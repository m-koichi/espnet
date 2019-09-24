#!/bin/bash

# Copyright 2019 Nagoya University (Koichi Miyazaki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch # pytorch only
stage=2         # start from 0 if you need to start from data preparation
stop_stage=2
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)

model=baseline

# FIXME: conda warning; unbound variable
#set -e
#set -u
train_dir=train_16k_mel64_rir
valid_dir=validation_16k_mel64_rir
eval_dir=eval_16k_mel64_rir


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    # git clone https://github.com/turpaultn/DCASE2019_task4.git
    # patch -p1 < sorted_label_index.patch
    # TODO: run in shell script. fix relative path problem
    #    cd DCASE2019_task4/baseline
    #    python ./download_data.py
    #    patch label_index.patch
    wget https://zenodo.org/record/2583796/files/Synthetic_dataset.zip -O ./DCASE2019_task4/dataset/Synthetic_dataset.zip
    unzip ./DCASE2019_task4/dataset/Synthetic_dataset.zip -d ./DCASE2019_task4/dataset
    rm ./DCASE2019_task4/dataset/Synthetic_dataset.zip
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    # Prepare scp files.
    python ./local/preprocess_data.py --train-dir ${train_dir} \
                                      --valid-dir ${valid_dir} \
                                      --eval-dir ${eval_dir}

    # Data augmentation with RIR reverberation.
    if [ ! -d "RIRS_NOISES" ]; then
        # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
        wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
        unzip rirs_noises.zip
        rm rirs_noises.zip
    fi
   . ./local/RIR_augment.sh data/${train_dir}

    # feature extraction for rir augmentation
#    for x in ${train_dir}; do
#        for f in text wav.scp utt2spk; do
#            sort data/${x}/${f} -o data/${x}/${f}
#        done
#        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
#        . ./local/make_fbank.sh data/${x} exp/make_fbank/${x} fbank
#        echo "feature"
#    done

    # feature extraction
    for x in ${train_dir} ${valid_dir} ${eval_dir}; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        . ./local/make_fbank.sh data/${x} exp/make_fbank/${x} fbank
        echo "feature"
    done

       # cp
    # merge data to json file
#     . ./local/data2json.sh --train_feat ./data/${train_dir}/feats.scp \
#                            --validation_feat ./data/${valid_dir}/feats.scp \
#                            --eval_feat ./data/${eval_dir}/feats.scp \
#                            --label ./DCASE2019_task4/dataset/metadata \
#                            --train_dir ${train_dir} \
#                            --valid_dir ${valid_dir} \
#                            --eval_dir ${eval_dir} \
#                            ./data

#     for label_type in synthetic weak; do
#         mv ${tmpdir}/output/label_${label_type}.scp ${tmpdir}/output/label_${label_type}.bak
#         cat ${tmpdir}/output/label_${label_type}.bak | while read meta; do
#             echo ${meta}
#             for i in `seq 1 ${augment_rep}`; do
#                 echo ${prefix}${i}_${meta}
#             done
#         done > ${tmpdir}/output/label_${label_type}.scp
#     done
#     echo "augmentation is done."                    
                           
    . ./local/data2json.sh --train_feat ./data/${train_dir}/feats.scp \
                           --validation_feat ./data/${valid_dir}/feats.scp \
                           --eval_feat ./data/${eval_dir}/feats.scp \
                           --label ./DCASE2019_task4/dataset/metadata \
                           --train_dir ${train_dir} \
                           --valid_dir ${valid_dir} \
                           --eval_dir ${eval_dir} \
                           --augment_rep 20 \
                           --prefix rvb \
                           ./data
     
                        
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    python ./sed/sed_train.py
fi
