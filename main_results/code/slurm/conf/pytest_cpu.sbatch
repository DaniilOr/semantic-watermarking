#!/bin/bash
#SBATCH --job-name=pytest_cpu
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1
/home/daniil.orel/anaconda3/envs/wmworld/bin/python -m pytest -x -q -m "not gpu" tests 2>&1 | tail -15
