#!/usr/bin/env bash
                           
                           
run_name=icassp_crnn_ssl


python -u sed/sed_train4.py --batch-size 64 \
                           --use-rir-augmentation False \
                           --use-specaugment False \
                           --use-post-processing False \
                           --opt adam \
                           --lr 0.001 \
                           --epochs 0 \
                           --iterations 10000 \
                           --log-interval 200 \
                           --model crnn \
                           --pooling-time-ratio 8 \
                           --loss-function BCE \
                           --noise-reduction False \
                           --pooling-operator attention \
                           --lr-scheduler cosine_annealing \
                           --T-max 100 \
                           --eta-min 1e-5 \
                           --train-data original \
                           --test-data original \
                           --n-frames 496 \
                           --dropout 0.5 \
                           --mels 64 \
                           --log-mels True \
                           --exp-mode SED \
                           --run-name $run_name \
                           --input-type 3 \
                           --ssl True \
                           | tee log/${run_name}.log