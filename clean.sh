#!/bin/bash
#########################
# clean the workspace
#########################

# kill experiment jobs in the cluster (Note tensorboard and jupyter jobs will not be killed)
scancel -n main -u $USER
# remove slurm outputs
rm slurm-*
# remove slurm log files 
rm *.pyc
# delete log files
# rm -r log
# delete checkpoints
rm -r checkpoints
# delete other log files
rm *.log
# clear the terminal screen
clear
