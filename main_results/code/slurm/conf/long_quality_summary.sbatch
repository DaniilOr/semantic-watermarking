#!/bin/bash
#SBATCH --job-name=long_quality
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
/home/daniil.orel/anaconda3/envs/wmworld/bin/python scripts/long_quality_summary.py runs/long_story
