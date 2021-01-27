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
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=0              # Processes starts from the specified stage.
stop_stage=5         # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=1                # The number of parallel jobs.
inference_nj=4      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# # Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw         # Feature type (raw or fbank_pitch).
audio_format=flac      # Audio format (only in feats_type=raw).
fs=16k                 # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Data augmentaion related
data_augmentation=true

# SED model related
sed_tag=    # Suffix to the result dir for sed model training.
sed_exp=    # Specify the direcotry path for SED experiment. If this option is specified, sed_tag is ignored.
sed_config= # Config for sed model training.
sed_args=   # Arguments for sed model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in sed config.
feats_normalize=global_mvn  # Normalizaton layer type
num_splits_sed=1   # Number of splitting for lm corpus

# Training related
train_config="" # Config for training.
train_args=""   # Arguments for training, e.g., "--max_epoch 1".
train_tag=""          # Suffix for training directory.


# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.best.pth       # Language modle path for decoding.
inference_sed_model=valid.acc.best.pth # SED model path for decoding.
                                    # e.g.
                                    # inference_sed_model=train.loss.best.pth
                                    # inference_sed_model=3epoch.pth
                                    # inference_sed_model=valid.acc.best.pth
                                    # inference_sed_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.


sed_audio_fold_length=500   # fold_length for speech data during SED training
sed_label_fold_length=500   # fold_length for text data during SED training

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    # --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type      # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format    # Audio format (only in feats_type=raw, default="${audio_format}").
    --fs              # Sampling rate (default="${fs}").

    # SED model related
    --sed_tag    # Suffix to the result dir for sed model training (default="${sed_tag}").
    --sed_exp    # Specify the direcotry path for SED experiment. If this option is specified, sed_tag is ignored (default="${sed_exp}").
    --sed_config # Config for sed model training (default="${sed_config}").
    --sed_args   # Arguments for sed model training, e.g., "--max_epoch 10" (default="${sed_args}").
                 # Note that it will overwrite args in sed config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_sed=1   # Number of splitting for lm corpus  (default="${num_splits_sed}").

    # Training related
    --trian_config # Config for training. (default="${train_config}").
    --train_args   # Arguments for training  (default="${train_args}").
                   # e.g., --train_args "--max_epoch 1"
                   # Note that it will overwrite args in train config
    --train_tag          # Suffix for training directory (default="${train_tag}")..

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding, e.g., "--lm_weight 0.1" (default="${inference_args}").
                       # Note that it will overwrite args in inference config.
    --inference_lm        # Language modle path for decoding (default="${inference_lm}").
    --inference_sed_model # SED model path for decoding (default="${inference_sed_model}").
    --download_model   # Download a model from Model Zoo and use it for decoding  (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set=    # Name of validation set used for monitoring/tuning network training (required).
    --test_sets=    # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified (required).

EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi


# Set tag for naming of model directory
if [ -z "${sed_tag}" ]; then
    if [ -n "${sed_config}" ]; then
        sed_tag="$(basename "${sed_config}" .yaml)_${feats_type}"
    else
        sed_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${sed_args}" ]; then
        sed_tag+="$(echo "${sed_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

if [ -z "${train_tag}" ]; then
    if [ -n "${sed_config}" ]; then
        train_tag="$(basename "${train_config}" .yaml)_${feats_type}"
    else
        train_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        train_tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi

    inference_tag+="_sed_model_$(echo "${inference_sed_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
sed_stats_dir="${expdir}/sed_stats_${feats_type}"
if [ -n "${speed_perturb_factors}" ]; then
    sed_stats_dir="${sed_stats_dir}_sp"
fi

# The directory used for training commands
if [ -z "${sed_exp}" ]; then
    sed_exp="${expdir}/sed_${sed_tag}"
fi
if [ -n "${speed_perturb_factors}" ]; then
    sed_exp="${sed_exp}_sp"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            # for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            for dset in "${train_set}" "${valid_set}"; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank_pitch ]; then
            log "[Require Kaldi] Stage 2: ${feats_type} extract: data/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
                utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                    | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank ]; then
            log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
            log "${feats_type} is not supported yet."
            exit 1

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
            # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # Generate dummy wav.scp to avoid error by copy_data_dir.sh
                <data/"${dset}"/cmvn.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
                utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"

                pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - | \
                    awk '{ print $2 }' | cut -d, -f2 > "${data_feats}${_suf}/${dset}/feats_dim"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                <"${data_feats}/org/${dset}/utt2num_samples" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples"
                <"${data_feats}/org/${dset}/wav.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                    | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                <"${data_feats}/org/${dset}/feats.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            # <"${data_feats}/org/${dset}/text" \
            #     awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done

        # shellcheck disable=SC2002
        # cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
    fi
else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _sed_train_dir="${data_feats}/${train_set}"
        _sed_valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: SED collect stats: train_set=${_sed_train_dir}, valid_set=${_sed_valid_dir}"

        _opts=
        if [ -n "${sed_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.sed_train --print_config --optim adam
            _opts+="--config ${sed_config} "
        fi

        _feats_type="$(<${_sed_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            _type=sound
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_sed_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${sed_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_sed_train_dir}/${_scp} wc -l)" "$(<${_sed_valid_dir}/${_scp} wc -l)")

        key_file="${_sed_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_sed_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${sed_stats_dir}/run.sh'. You can resume the process from stage 9 using this script"
        mkdir -p "${sed_stats_dir}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${sed_stats_dir}/run.sh"; chmod +x "${sed_stats_dir}/run.sh"

        # 3. Submit jobs
        log "SED collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        
        # ${python} -m espnet2.bin.sed_train \
        # shellcheck disable=SC2086
        # ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            # ${python} -m pyscripts.sed_train \
            #     --collect_stats true \
            #     --model_conf "conf/model.yaml" \
            #     --train_data_path_and_name_and_type "${_sed_train_dir}/${_scp},audio,${_type}" \
            #     --valid_data_path_and_name_and_type "${_sed_valid_dir}/${_scp},audio,${_type}" \
            #     --train_shape_file "${_logdir}/train.JOB.scp" \
            #     --valid_shape_file "${_logdir}/valid.JOB.scp" \
            #     --output_dir "${_logdir}/stats.JOB" \
            #     ${_opts} ${sed_args}

        ${python} -m pyscripts.sed_train \
                --collect_stats true \
                --use_preprocessor true \
                --label_list "label_list.txt" \
                --train_data_path_and_name_and_type "${_sed_train_dir}/${_scp},audio,${_type}" \
                --train_data_path_and_name_and_type "${_sed_train_dir}/labels.txt,label,text " \
                --valid_data_path_and_name_and_type "${_sed_valid_dir}/${_scp},audio,${_type}" \
                --valid_data_path_and_name_and_type "${_sed_valid_dir}/labels.txt,label,text " \
                --train_shape_file "${_logdir}/train.1.scp" \
                --valid_shape_file "${_logdir}/valid.1.scp" \
                --output_dir "${_logdir}/stats.1" \
                ${_opts} ${sed_args}

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${sed_stats_dir}"
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _sed_train_dir="${data_feats}/${train_set}"
        _sed_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5 SED Training: train_set=${_sed_train_dir}, valid_set=${_sed_valid_dir}"

        _opts=
        if [ -n "${sed_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.sed_train --print_config --optim adam
            _opts+="--config ${sed_config} "
        fi

        _feats_type="$(<${_sed_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            _type=sound
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${sed_audio_fold_length}"
            _input_size="$(<${_sed_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${sed_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_sed}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${sed_stats_dir}/splits${num_splits_sed}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_sed_train_dir}/${_scp}" \
                      "${sed_stats_dir}/train/audio_shape" \
                  --num_splits "${num_splits_sed}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},audio,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/audio_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_sed_train_dir}/${_scp},audio,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_sed_train_dir}/labels.txt,label,text "
            _opts+="--train_shape_file ${sed_stats_dir}/train/audio_shape "
        fi

        log "Generate '${sed_exp}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${sed_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${sed_exp}/run.sh"; chmod +x "${sed_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "SED training started... log: '${sed_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${sed_exp})"
        else
            jobname="${sed_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        # ${python} -m espnet2.bin.launch \
        #     --cmd "${cuda_cmd} --name ${jobname}" \
        #     --log "${sed_exp}"/train.log \
        #     --ngpu "${ngpu}" \
        #     --num_nodes "${num_nodes}" \
        #     --init_file_prefix "${sed_exp}"/.dist_init_ \
        #     --multiprocessing_distributed true -- \
            # ${python} -m espnet2.bin.sed_train \
            # ${python} -m pyscripts.sed_train \
            #     --use_preprocessor true \
            #     --bpemodel "${bpemodel}" \
            #     --token_type "${token_type}" \
            #     --token_list "${token_list}" \
            #     --non_linguistic_symbols "${nlsyms_txt}" \
            #     --cleaner "${cleaner}" \
            #     --g2p "${g2p}" \
            #     --valid_data_path_and_name_and_type "${_sed_valid_dir}/${_scp},speech,${_type}" \
            #     --valid_data_path_and_name_and_type "${_sed_valid_dir}/text,text,text" \
            #     --valid_shape_file "${sed_stats_dir}/valid/speech_shape" \
            #     --valid_shape_file "${sed_stats_dir}/valid/text_shape.${token_type}" \
            #     --resume true \
            #     --fold_length "${_fold_length}" \
            #     --fold_length "${sed_text_fold_length}" \
            #     --output_dir "${sed_exp}" \
            #     ${_opts} ${sed_args}
        ${python} -m pyscripts.sed_train \
            --use_preprocessor true \
            --label_list "label_list.txt" \
            --num_workers 0 \
            --num_att_plot 0 \
            --ngpu "${ngpu}" \
            --valid_data_path_and_name_and_type "${_sed_valid_dir}/${_scp},audio,${_type}" \
            --valid_data_path_and_name_and_type "${_sed_valid_dir}/labels.txt,label,text " \
            --valid_shape_file "${sed_stats_dir}/valid/audio_shape" \
            --resume true \
            --fold_length "${sed_audio_fold_length}" \
            --output_dir "${sed_exp}" \
            ${_opts} ${sed_args}
    fi
else
    log "Skip the training stages"
fi

#TODO: Following has not been checked yet
if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    sed_exp="${expdir}/${download_model}"
    mkdir -p "${sed_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${sed_exp}/config.txt"

    # Get the path of each file
    _sed_model_file=$(<"${sed_exp}/config.txt" sed -e "s/.*'sed_model_file': '\([^']*\)'.*$/\1/")
    _sed_train_config=$(<"${sed_exp}/config.txt" sed -e "s/.*'sed_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_sed_model_file}" "${sed_exp}"
    ln -sf "${_sed_train_config}" "${sed_exp}"
    inference_sed_model=$(basename "${_sed_model_file}")

    if [ "$(<${sed_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${sed_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${sed_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Decoding: training_dir=${sed_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        # 2. Generate run.sh
        log "Generate '${sed_exp}/${inference_tag}/run.sh'. You can resume the process from stage 11 using this script"
        mkdir -p "${sed_exp}/${inference_tag}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${sed_exp}/${inference_tag}/run.sh"; chmod +x "${sed_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${sed_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                _type=sound
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/sed_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/sed_inference.JOB.log \
                ${python} -m espnet2.bin.sed_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --sed_train_config "${sed_exp}"/config.yaml \
                    --sed_model_file "${sed_exp}"/"${inference_sed_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done
        done
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${sed_exp}/${sed_exp##*/}_${inference_sed_model%.*}.zip"
if ! "${skip_upload}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Pack model: ${packed_model}"

        _opts=
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${sed_stats_dir}/train/feats_stats.npz "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack sed \
            --sed_train_config "${sed_exp}"/config.yaml \
            --sed_model_file "${sed_exp}"/"${inference_sed_model}" \
            ${_opts} \
            --option "${sed_exp}"/RESULTS.md \
            --outpath "${packed_model}"
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Upload model to Zenodo: ${packed_model}"

        # To upload your model, you need to do:
        #   1. Sign up to Zenodo: https://zenodo.org/
        #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
        #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="
git checkout $(git show -s --format=%H)"

        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/sed1/ -> foo/sed1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/sed1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${sed_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${sed_exp}"/RESULTS.md)</code></pre></li>
<li><strong>sed config</strong><pre><code>$(cat "${sed_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by youself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${sed_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
