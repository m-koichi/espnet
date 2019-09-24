#!/bin/bash

# Copyright 2019 Nagoya University (Koichi Miyazaki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=32
cmd=run.pl
train_feat="" # feat.scp for training
validation_feat="" # feat.scp for validation
eval_feat="" # feat.scp for evaluation
label="" # metadata directory e.g. ./DCASE2019_task4/dataset/metadata
verbose=0
augment_rep=20
prefix=rvb
filetype=""
preprocess_conf=""

train_dir=train_16k_mel64_rir
valid_dir=validation_16k_mel64_rir
eval_dir=eval_16k_mel64_rir

. utils/parse_options.sh

if [ $# != 1 ]; then
    cat << EOF 1>&2
Usage: $0 <data-dir>
e.g. $0 data/train data/lang_1char/train_units.txt
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --train_feat <feat-scp>                          # feat.scp for training
  --validation_feat <feat-scp>                     # feat.scp for validation
  --label                                          # metadata directory
  --oov <oov-word>                                 # Default: <unk>
  --out <outputfile>                               # If omitted, write in stdout
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
    exit 1;
fi

# FIXME: conda warning; unbound variable
#set -euo pipefail

dir=$1
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

# 1. Create scp files for inputs
#   These are not necessary for decoding mode, and make it as an option
mkdir -p ${tmpdir}/input
if [ -n "${train_feat}" ]; then
    for label_type in synthetic unlabel_in_domain weak; do
        grep ${label_type} ${train_feat} > ${tmpdir}/input/feats_${label_type}.scp
        feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
                         --filetype "${filetype}" \
                         --preprocess-conf "${preprocess_conf}" \
                         --verbose ${verbose} ${tmpdir}/input/feats_${label_type}.scp ${tmpdir}/input/shape_${label_type}.scp
    done
fi

if [ -n "${validation_feat}" ]; then
    for label_type in eval_dcase2018 test_dcase2018 validation; do
        grep ^${label_type} ${validation_feat} > ${tmpdir}/input/feats_${label_type}.scp
        feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
                         --filetype "${filetype}" \
                         --preprocess-conf "${preprocess_conf}" \
                         --verbose ${verbose} ${tmpdir}/input/feats_${label_type}.scp ${tmpdir}/input/shape_${label_type}.scp
    done
fi

if [ -n "${eval_feat}" ]; then
    for label_type in eval_dcase2019; do
        grep ^${label_type} ${eval_feat} > ${tmpdir}/input/feats_${label_type}.scp
        feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
                         --filetype "${filetype}" \
                         --preprocess-conf "${preprocess_conf}" \
                         --verbose ${verbose} ${tmpdir}/input/feats_${label_type}.scp ${tmpdir}/input/shape_${label_type}.scp
    done
fi


# 2. Create scp files for outputs
mkdir -p ${tmpdir}/output

if [ -n "${label}" ]; then
    for x in train validation; do
        if [ ${x} = train ]; then
            for label_type in synthetic unlabel_in_domain weak; do
                csv=${label}/${x}/${label_type}.csv
                if [ ${label_type} = synthetic ]; then
                    for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
                        echo -n ${label_type}-$(basename $id .wav)
                        cat $csv | grep ^${id} | awk '{printf " %s %s %s", $2, $3, $4}'
                        echo ""
                    done > ${tmpdir}/output/label_${label_type}.scp
                else
                    for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
                        echo -n ${label_type}-$(basename $id .wav)
                        cat $csv | grep ^${id} | awk '{printf " %s", $2}'
                        echo ""
                    done > ${tmpdir}/output/label_${label_type}.scp
                fi
            done
        elif [ ${x} = validation ]; then
            for label_type in eval_dcase2018 test_dcase2018 validation; do
                csv=${label}/${x}/${label_type}.csv
                for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
                    echo -n ${label_type}-$(basename $id .wav)
                    cat $csv | grep ^${id} | awk '{printf " %s %s %s", $2, $3, $4}'
                    echo ""
                done > ${tmpdir}/output/label_${label_type}.scp
            done

#        elif [ ${x} = eval ]; then
#            for label_type in eval_dcase2019; do
#                csv=${label}/${x}/${label_type}.csv
#                for id in $(tail -n +2 ${csv} | awk '{print $1}' | uniq); do
#                    echo -n ${label_type}-$(basename $id .wav)
#                    cat $csv | grep ^${id} | awk '{printf " %s %s %s", $2, $3, $4}'
#                    echo ""
#                done > ${tmpdir}/output/label_${label_type}.scp
#            done
        fi
    done
fi

echo "output done"

# perform data augmentation
echo ${augment_rep}
echo ${prefix}
if [ ${augment_rep} -gt 0 ] && [ -n "${prefix}" ]; then
    echo "start augmentation"
    for label_type in synthetic weak; do
        mv ${tmpdir}/output/label_${label_type}.scp ${tmpdir}/output/label_${label_type}.bak
        cat ${tmpdir}/output/label_${label_type}.bak | while read meta; do
            echo ${meta}
            for i in `seq 1 ${augment_rep}`; do
                echo ${prefix}${i}-${meta}
            done
        done > ${tmpdir}/output/label_${label_type}.scp
    done
    echo "augmentation is done."
fi
cp ${tmpdir}/input/feats_synthetic.scp ./feats_tmp.scp
cp ${tmpdir}/output/label_synthetic.scp ./label_tmp.scp
# 3. Merge scp files into a JSON file
# for x in train validation; do
for x in ${train_dir} ${valid_dir} ${eval_dir}; do
# for x in train_aug; do
    # if [ ${x} = train ]; then
   if [ ${x} = ${train_dir} ]; then
        for label_type in synthetic unlabel_in_domain weak; do
            opts=""
            opts+="--input-scps "
            opts+="feats:${tmpdir}/input/feats_${label_type}.scp "
            opts+="shape:${tmpdir}/input/shape_${label_type}.scp:shape "
            if [ ${label_type} != unlabel_in_domain ]; then
                sort ${tmpdir}/input/feats_${label_type}.scp -o ${tmpdir}/input/feats_${label_type}.scp
                sort ${tmpdir}/output/label_${label_type}.scp -o ${tmpdir}/output/label_${label_type}.scp
                opts+="--output-scps "
                opts+="label:${tmpdir}/output/label_${label_type}.scp "
            fi
            ./local/merge_scp2json.py --verbose ${verbose} ${opts} > ./data/${x}/data_${label_type}.json
        done

   elif [ ${x} = ${valid_dir} ]; then
        for label_type in eval_dcase2018 test_dcase2018 validation; do
            opts=""
            opts+="--input-scps "
            opts+="feats:${tmpdir}/input/feats_${label_type}.scp "
            opts+="shape:${tmpdir}/input/shape_${label_type}.scp:shape "
            if [ ${label_type} != unlabel_in_domain ]; then
                sort ${tmpdir}/input/feats_${label_type}.scp -o ${tmpdir}/input/feats_${label_type}.scp
                sort ${tmpdir}/output/label_${label_type}.scp -o ${tmpdir}/output/label_${label_type}.scp
                opts+="--output-scps "
                opts+="label:${tmpdir}/output/label_${label_type}.scp "
            fi
            cp ${tmpdir}/input/feats_${label_type}.scp ./data/feats.scp
            cp ${tmpdir}/output/label_${label_type}.scp ./data/labels.scp
            ./local/merge_scp2json.py --verbose ${verbose} ${opts} > ./data/${x}/data_${label_type}.json
        done

   elif [ ${x} = ${eval_dir} ]; then
        for label_type in eval_dcase2019; do
            opts=""
            opts+="--input-scps "
            opts+="feats:${tmpdir}/input/feats_${label_type}.scp "
            opts+="shape:${tmpdir}/input/shape_${label_type}.scp:shape "
#            if [ ${label_type} != unlabel_in_domain ]; then
#                sort ${tmpdir}/input/feats_${label_type}.scp -o ${tmpdir}/input/feats_${label_type}.scp
#                sort ${tmpdir}/output/label_${label_type}.scp -o ${tmpdir}/output/label_${label_type}.scp
#                opts+="--output-scps "
#                opts+="label:${tmpdir}/output/label_${label_type}.scp "
#            fi
#            cp ${tmpdir}/input/feats_${label_type}.scp ./data/feats.scp
#            cp ${tmpdir}/output/label_${label_type}.scp ./data/labels.scp
            ./local/merge_scp2json.py --verbose ${verbose} ${opts} > ./data/${x}/data_${label_type}.json
        done
   fi
done

rm -fr ${tmpdir}
