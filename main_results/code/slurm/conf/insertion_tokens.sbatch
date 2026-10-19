#!/bin/bash
#SBATCH --job-name=insertion_tokens
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
/home/daniil.orel/anaconda3/envs/wmworld/bin/python scripts/insertion_tokens.py
