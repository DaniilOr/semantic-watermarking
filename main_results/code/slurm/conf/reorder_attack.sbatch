#!/bin/bash
#SBATCH --job-name=reorder_attack
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
/home/daniil.orel/anaconda3/envs/wmworld/bin/python scripts/reorder_attack.py --run runs/conf_cross --arms base,wm_c4
