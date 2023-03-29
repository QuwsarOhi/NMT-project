#!/bin/bash
####### Reserve computing resources #############
#SBATCH --job-name=nmt-train
#SBATCH --output=out.txt
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-v100
#SBATCH --time=04:50:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

####### Set environment variables ###############

####### Run your script #########################
python train.py

####### run the code by: sbatch job-script.slurm #######
####### kill process by: scancel jobid #######
####### check process status by: squeue --job jobid #######