#!/bin/bash

# Bash script for running several EEGNet models consecutively

# Make fmax the first argument
fmax=$1

if [ -z "$fmax" ]
then
    echo "Please specify fmax as the first argument"
    exit 1
fi

python eegnet_psd.py --online --use_rfe --rfe_method SVC --fmax $fmax
python eegnet_psd.py --online --use_rfe --rfe_method LogisticRegression --fmax $fmax
python eegnet_psd.py --fmax $fmax
python eegnet_psd.py --online --fmax $fmax