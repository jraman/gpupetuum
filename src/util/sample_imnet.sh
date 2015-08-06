#!/bin/bash

NUMSAMPLES=$1

[ "x$NUMSAMPLES" = "x" ] && { echo "Need NUMSAMPLES"; exit 1; }
[ $(( $NUMSAMPLES + 0 )) = $NUMSAMPLES ] || { echo "NUMSAMPLES must be a number"; exit 1; }

DATADIR="../imnet_data"
LABELFILE="$DATADIR/0_label.txt"
DATAFILE="$DATADIR/imnet_0.bin"


SELECTLABELFILE="$DATADIR/label_select.n${NUMSAMPLES}.txt"
OUTFILE="$DATADIR/imnet_sample.n${NUMSAMPLES}.pkl"
OUTLABELFILE="$DATADIR/label_sample.n${NUMSAMPLES}.txt"

set -x

head -n $NUMSAMPLES $LABELFILE > $SELECTLABELFILE

python pickle_llc.py $DATAFILE $LABELFILE $OUTFILE $SELECTLABELFILE $OUTLABELFILE
