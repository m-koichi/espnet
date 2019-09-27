#!/usr/bin/env bash

run_name=0925013
model=transformer


python -u sed/saigen3.py --run-name $run_name \
                         --averaged False \
                         | tee exp3/${run_name}/pp_tuning.log
                         
                         
# Get averaged model
python -u sed/average_checkpoint.py --run-name ${run_name} \
                                    --model ${model}
                         
python -u sed/saigen3.py --run-name $run_name \
                         --averaged True \
                         | tee exp3/${run_name}/pp_tuning_avg.log