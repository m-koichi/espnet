#!/bin/bash

python -u sed/sed_train3.py --batch-size 16 \
                           --ngpu 1 \
                           --seed 1 \
                           --tensorboard-dir tensorboard \
                           --use-rir-augmentation False \
                           --use-specaugment False \
                           --use-post-processing False \
                           --add-noise False \
                           --model transformer_ema_fixed \
                           --opt noam \
                           --epochs 300 \
                           --iterations 30000 \
                           --log-interval 200 \
                           --pooling-time-ratio 1 \
                           --loss-function BCE \
                           --pooling-operator attention \
                           --train-data original \
                           --test-data original \
                           --n-frames 500 \
                           --dropout 0.2 \
                           --dropout-rate 0.2 \
                           --mels 128 \
                           --log-mels True \
                           --warm-start True \
                           --transformer-init pytorch \
                           --transformer-input-layer linear \
                           --transformer-attn-dropout-rate 0.2 \
                           --transformer-lr 0.01 \
                           --transformer-warmup-steps 4000 \
                           --transformer-length-normalized-loss True \
                           --adim 128 \
                           --aheads 4 \
                           --elayers 3 \
                           --eunits 512 \
                           --accum-grad 4 \
                           --input-layer-type 3



#python average_checkpoint.py --snapshots model \
#                             --out average.pth \
#                             --num 10 \
#                             --backend pytorch
