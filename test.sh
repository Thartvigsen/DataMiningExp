#!/bin/bash
################################
# How to use this script?
# This script is used for unit tests
# in Cluster Head Node terminal, type: ./test.sh [filename]
# the [filename] is the file that you want to run unit tests, for example, model_test.py
################################

export PYTHONPATH=./env/bin/python
source ./env/bin/activate
#srun --pty -t 5 --gres=gpu:2 nosetests -v --nologcapture -s $1
srun --pty -t 5 --gres=gpu:0 --mem=24G nosetests $1 -v --nologcapture
# -s $1
#srun --pty -t 5 --gres=gpu:2 --mem=64G python main.py --taskid=$41
rm *.pyc

