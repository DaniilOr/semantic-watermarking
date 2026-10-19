#!/bin/bash
#SBATCH --job-name=pref_long
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# preference-only judge on the long-text run and the first span run, for consistency with the other numbers
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/long_c --run runs/spans --workers 8
echo "end=$(date)"
