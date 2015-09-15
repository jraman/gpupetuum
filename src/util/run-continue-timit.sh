#!/bin/bash -x

# Typical usage:
# lockrun -L=/tmp/timit.lock --wait --sleep=60 -- <thisfile> <begin_epoch> <end_epoch>
# Example:
# lockrun -L=/tmp/timit.lock --wait --sleep=60 -- run-continue-timit.sh 1100 1200

# python theanodbn/runners/run_dbn_rambatch.py -c conf/timit/timit_c2001_l6_2048.py --continue-run --finetune-lr=0.6 --finetuned-model-file=../model/timit/best_eta.ep300/timit.finetune.lr0.6.ep300.pkl --finetune-epoch-start=200 --finetune-training-epochs=300 --start-model-file=../model/timit/best_eta.ep200/timit.finetune.lr0.6.ep200.pkl 2>&1 | tee -a ../log/timit/best_eta.ep300/timit_c2001_l6_2048.0.6.log

[ "x$1" = "x" ] && { echo "Error: No arg1"; exit 1; }
[ "x$2" = "x" ] && { echo "Error: No arg2"; exit 1; }
[ $1 -ge $2 ] && { echo "Error: arg1 is greater-than-or-equal-to arg2"; exit 1; }

BEGIN=$1
END=$2

mkdir -p ../log/timit/best_eta.ep${END}
mkdir -p ../model/timit/best_eta.ep${END}

LOGFILE="../log/timit/best_eta.ep${END}/timit_c2001_l6_2048.0.6.log"
touch $LOGFILE

python theanodbn/runners/run_dbn_rambatch.py -c conf/timit/timit_c2001_l6_2048.py --continue-run --finetune-lr=0.6 --finetuned-model-file=../model/timit/best_eta.ep${END}/timit.finetune.lr0.6.ep${END}.pkl --finetune-epoch-start=${BEGIN} --finetune-training-epochs=${END} --start-model-file=../model/timit/best_eta.ep${BEGIN}/timit.finetune.lr0.6.ep${BEGIN}.pkl 2>&1 | tee -a ${LOGFILE}

chmod -w ../model/timit/best_eta.ep${END}/timit.finetune.lr0.6.ep${END}.pkl
chmod -w ../log/timit/best_eta.ep${END}/timit_c2001_l6_2048.0.6.log

