#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sed_config=conf/train_conf.yaml

./sed.sh \
    --stage 10 \
    --stop_stage 10 \
    --sed_tag "${sed_tag}" \
    --sed_config "${sed_config}" \
    --train_set train \
    --valid_set dev 