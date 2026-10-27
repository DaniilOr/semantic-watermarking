#!/bin/bash
#SBATCH --job-name=pmark_check
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
/home/daniil.orel/anaconda3/envs/wmworld/bin/python scripts/pmark_check.py
echo "exit=$?"
