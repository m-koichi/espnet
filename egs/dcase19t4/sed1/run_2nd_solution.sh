#!/usr/bin/env bash

if [[ `hostname` == *'.lineaws.com' ]]; 
then
    export CUDA_VISIBLE_DEVICES=2
fi

python -u sed/sed_train4.py --batch-size 32 \
                           --use-rir-augmentation False \
                           --use-specaugment False \
                           --use-post-processing False \
                           --opt adam \
                           --lr 0.001 \
                           --epochs 300 \
                           --iterations 30000 \
                           --log-interval 200 \
                           --model crnn \
                           --pooling-time-ratio 1 \
                           --loss-function BCE \
                           --noise-reduction False \
                           --pooling-operator attention \
                           --lr-scheduler cosine_annealing \
                           --T-max 100 \
                           --eta-min 1e-5 \
                           --train-data original \
                           --test-data original \
                           --n-frames 864 \
                           --dropout 0.5 \
                           --mels 128 \
                           --log-mels True
