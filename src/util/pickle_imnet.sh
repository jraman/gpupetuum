#!/bin/bash

DATADIR=../imnet_data
python pickle_llc.py -i ${DATADIR}/imnet_0.bin -l ${DATADIR}/0_label.txt -o ${DATADIR}/imnet_0.pkl -n 21504
