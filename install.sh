#!/bin/bash

#------------------------------------ 
# This script install the required modules and packages
# Please run this script when you are using the system for the first time
# How to use this script?
#   In the terminal, type: ./install.sh
#------------------------------------ 

#------------------------------
# load slurm modules to support GPU computation 

# remove all modules previously loaded
#module purge
### add modules 
#module add slurm cm-ml-pythondeps cudnn/gcc-4.8.5/6.0 openblas/dynamic cuda80/toolkit hdf5_18
### load modules when user login (initial load list)

module initadd slurm

# Note: if you get errors by running codes after installing the modules, log out and re-login the system, try again.

#------------------------------
# install virtual environment tool 

# create virtual environment in folder env/ 
virtualenv ./env

# activate virtual environment from folder env/ 
source env/bin/activate
export PYTHONPATH=./env/bin/python2.7

# install python packages
pip install -r requirements.txt
