#!/bin/bash

python -u sed/sed_train.py --batch-size 16 \
                           --use-rir-augmentation False \
                           --use-specaugment False \
                           --use-post-processing False \
                           --opt adam \
                           --lr 0.001 \
                           --epochs 100\
                           --iterations 30000 \
                           --log-interval 200 \
                           --model crnn_after \
                           --pooling-time-ratio 1 \
                           --loss-function BCE \
                           --noise-reduction False \
                           --pooling-operator attention \
                           --lr-scheduler cosine_annealing \
                           --T-max 100 \
                           --eta-min 1e-5 \
                           --train-data original \
                           --test-data original \
                           --n-frames 500 \
                           --dropout 0.5 \
                           --mels 64 \
                           --log-mels False
