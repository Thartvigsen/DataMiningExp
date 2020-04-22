#!/bin/bash
#SBATCH -p short
#SBATCH -N 1                      # number of nodes
#SBATCH -n 4                      # number of cores
#SBATCH --mem=1GB               # memory pool for all cores
#SBATCH -t 0-24:00                # time (D-HH:MM)
#SBATCH --checkpoint=5
#SBATCH --checkpoint-dir=checkpoints
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH --job-name=main
#SBATCH -o slurm-main-output%a    # STDOUT
#SBATCH -e slurm-main-error%a     # STDERR
##SBATCH --mail-type=END
##SBATCH --mail-user=twhartvigsen@wpi.edu

#python main.py --taskid=${SLURM_ARRAY_TASK_ID} #--num_gpu=1
python run_models.py --taskid=${SLURM_ARRAY_TASK_ID} #--num_gpu=1
