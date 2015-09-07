#!/bin/bash -e

for lr in 0.001 0.003 0.006 0.01 0.03 0.06 0.1 0.3 0.6 1.0 3.0
do
    echo "lr=$lr"
    set -x
    python theanodbn/runners/run_dbn_rambatch.py -c conf/timit/timit_c2001_l6_2048.py --finetune-lr=${lr} 2>&1 | tee ../log/timit/timit_c2001_l6_2048.${lr}.log
    set +x
done
