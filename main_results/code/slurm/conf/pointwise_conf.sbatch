#!/bin/bash
#SBATCH --job-name=pointwise_conf
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
# ABSOLUTE quality on the HELD-OUT texts, separate from the pairwise win rate: pointwise judge + test against clean variation
$PY scripts/api_pointwise.py --config $CFG --run runs/conf_cross --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/conf_pos --arms base --workers 8
$PY scripts/quality_vs_clean.py --clean runs/conf_cross:base --clean runs/conf_pos:base --marked runs/conf_cross --out runs/quality_vs_clean_conf.json
echo "end=$(date)"
