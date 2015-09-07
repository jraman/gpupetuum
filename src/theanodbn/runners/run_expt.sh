#!/bin/bash -e

# graceful stop if this file exists
STOPFILE=/tmp/gracefulstop

for lr in 0.6 0.8 1.0 1.2
# for lr in 0.1 0.4 0.6 0.8 1.0 1.2
# for lr in 0.4 0.5 0.7 0.8 0.9 1.1 1.2
# for lr in 0.001 0.003 0.006 0.01 0.03 0.06 0.1 0.3 0.6 1.0 3.0
do
    if [ -e $STOPFILE ]; then
        echo "Found ${STOPFILE}.  Stopping."
        exit 0
    fi
    ftmodelfile="../model/timit/150907/timit.finetune.lr${lr}.ep1000.pkl"
    set -x
    python theanodbn/runners/run_dbn_rambatch.py \
        -c conf/timit/timit_c2001_l6_2048.py \
        --finetune-lr=${lr} \
        --finetuned-model-file=${ftmodelfile} \
        2>&1 | tee ../log/timit/timit_c2001_l6_2048.${lr}.log
    set +x
done
