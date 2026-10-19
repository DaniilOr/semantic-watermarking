#!/bin/bash
#SBATCH --job-name=pref_only
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
# PURE preference: two anonymous responses in both orders, one question -- which is better?  (flaws are judged separately)
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/conf_cross --workers 8
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/tradeoff --workers 8
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/ifeval_cross --workers 8
echo "end=$(date)"
