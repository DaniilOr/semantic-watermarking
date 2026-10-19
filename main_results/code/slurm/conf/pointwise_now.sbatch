#!/bin/bash
#SBATCH --job-name=pointwise_now
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
# ABSOLUTE quality, one text at a time (kept separate from the pairwise win rate): texts that exist right now
$PY scripts/api_pointwise.py --config $CFG --run runs/tradeoff --run runs/cross_llama --workers 8
echo "end=$(date)"
