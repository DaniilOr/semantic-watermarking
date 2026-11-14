#!/bin/bash
#SBATCH --job-name=lr_eval
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld; export RUNS_ROOT=/nfs-stor/daniil.orel/home_moved/wmworld_runs
/home/daniil.orel/anaconda3/envs/wmworld/bin/python scripts/lr_detect.py eval
