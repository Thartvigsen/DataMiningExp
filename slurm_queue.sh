#!/bin/bash
################################
# Check the status of the job queue in the cluster
# How to use this script?
# in Cluster Head Node terminal, type: ./slurm_queue.sh
################################

squeue -u $USER
